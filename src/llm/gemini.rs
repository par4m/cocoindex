use async_trait::async_trait;
use crate::llm::{LlmGenerationClient, LlmSpec, LlmGenerateRequest, LlmGenerateResponse, ToJsonSchemaOptions, OutputFormat};
use anyhow::{Result, anyhow};
use serde_json;
use reqwest::Client as HttpClient;
use serde_json::Value;

pub struct Client {
    model: String,
}

impl Client {
    pub async fn new(spec: LlmSpec) -> Result<Self> {
        if std::env::var("GEMINI_API_KEY").is_err() {
            anyhow::bail!("GEMINI_API_KEY environment variable must be set");
        }
        Ok(Self {
            model: spec.model,
        })
    }
}

// Recursively remove all `additionalProperties` fields from a JSON value
fn remove_additional_properties(value: &mut Value) {
    match value {
        Value::Object(map) => {
            map.remove("additionalProperties");
            for v in map.values_mut() {
                remove_additional_properties(v);
            }
        }
        Value::Array(arr) => {
            for v in arr {
                remove_additional_properties(v);
            }
        }
        _ => {}
    }
}

#[async_trait]
impl LlmGenerationClient for Client {
    async fn generate<'req>(
        &self,
        request: LlmGenerateRequest<'req>,
    ) -> Result<LlmGenerateResponse> {
        // Compose the prompt/messages
        let contents = vec![serde_json::json!({
            "role": "user",
            "parts": [{ "text": request.user_prompt }]
        })];

        // Optionally add system prompt
        let mut system_instruction = None;
        if let Some(system) = request.system_prompt {
            system_instruction = Some(serde_json::json!({
                "parts": [{ "text": system }]
            }));
        }

        // Prepare payload
        let mut payload = serde_json::json!({ "contents": contents });
        if let Some(system) = system_instruction {
            payload["systemInstruction"] = system;
        }

        // If structured output is requested, add schema and responseMimeType
        if let Some(OutputFormat::JsonSchema { schema, .. }) = &request.output_format {
            let mut schema_json = serde_json::to_value(schema)?;
            remove_additional_properties(&mut schema_json);
            payload["generationConfig"] = serde_json::json!({
                "responseMimeType": "application/json",
                "responseSchema": schema_json
            });
        }

        let api_key = std::env::var("GEMINI_API_KEY")
            .map_err(|_| anyhow!("GEMINI_API_KEY environment variable must be set"))?;
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            self.model, api_key
        );

        let client = HttpClient::new();
        let resp = client.post(&url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| anyhow!("HTTP error: {e}"))?;

        let resp_json: Value = resp.json().await.map_err(|e| anyhow!("Invalid JSON: {e}"))?;

        if let Some(error) = resp_json.get("error") {
            return Err(anyhow!("Gemini API error: {:?}", error));
        }
        let text = resp_json["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .unwrap_or("")
            .to_string();

        Ok(LlmGenerateResponse { text })
    }

    fn json_schema_options(&self) -> ToJsonSchemaOptions {
        ToJsonSchemaOptions {
            fields_always_required: false,
            supports_format: false,
            extract_descriptions: false,
            top_level_must_be_object: true,
        }
    }
}