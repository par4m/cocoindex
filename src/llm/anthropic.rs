use async_trait::async_trait;
use crate::llm::{LlmGenerationClient, LlmSpec, LlmGenerateRequest, LlmGenerateResponse, ToJsonSchemaOptions, OutputFormat};
use anyhow::{Result, bail, Context};
use serde_json::Value;
use crate::api_bail;
use urlencoding::encode;

pub struct Client {
    model: String,
    api_key: String,
    client: reqwest::Client,
}

impl Client {
    pub async fn new(spec: LlmSpec) -> Result<Self> {
        let api_key = match std::env::var("ANTHROPIC_API_KEY") {
            Ok(val) => val,
            Err(_) => api_bail!("ANTHROPIC_API_KEY environment variable must be set"),
        };
        Ok(Self {
            model: spec.model,
            api_key,
            client: reqwest::Client::new(),
        })
    }
}

#[async_trait]
impl LlmGenerationClient for Client {
    async fn generate<'req>(
        &self,
        request: LlmGenerateRequest<'req>,
    ) -> Result<LlmGenerateResponse> {
        // Compose the prompt/messages
        let mut messages = vec![serde_json::json!({
            "role": "user",
            "content": request.user_prompt
        })];
        if let Some(system) = request.system_prompt {
            messages.insert(0, serde_json::json!({
                "role": "system",
                "content": system
            }));
        }

        let mut payload = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096
        });

        // If structured output is requested, add schema
        if let Some(OutputFormat::JsonSchema { schema, .. }) = &request.output_format {
            let schema_json = serde_json::to_value(schema)?;
            payload["tools"] = serde_json::json!([
                { "type": "json_object", "parameters": schema_json }
            ]);
        }

        let url = "https://api.anthropic.com/v1/messages";

        let encoded_api_key = encode(&self.api_key);
        let resp = self.client
            .post(url)
            .header("x-api-key", encoded_api_key.as_ref())
            .json(&payload)
            .send()
            .await
            .context("HTTP error")?;

        let resp_json: Value = resp.json().await.context("Invalid JSON")?;

        if let Some(error) = resp_json.get("error") {
            bail!("Anthropic API error: {:?}", error);
        }
        let mut resp_json = resp_json;
        let text = match &mut resp_json["content"][0]["text"] {
            Value::String(s) => std::mem::take(s),
            _ => bail!("No text in response"),
        };

        Ok(LlmGenerateResponse {
            text,
        })
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
