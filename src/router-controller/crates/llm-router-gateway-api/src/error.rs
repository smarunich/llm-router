//! Error
use std::convert::Infallible;

#[derive(Debug, thiserror::Error)]
pub enum GatewayApiError {
    #[error("Generic Error: {0}")]
    Generic(String),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    InvalidUri(#[from] http::uri::InvalidUri),

    #[error(transparent)]
    InvalidHeaderValue(#[from] http::header::InvalidHeaderValue),

    #[error(transparent)]
    Yaml(#[from] serde_yaml::Error),

    #[error("OpenAI API Error: {0}")]
    OpenAI(String),

    #[error("Could not parse URI: {0}")]
    Uri(String),

    #[error("Internal Server Error")]
    InternalServer,

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Policy not found: {0}")]
    PolicyNotFound(String),

    #[error("No routing strategy specified")]
    NoRoutingStrategy,

    #[error(transparent)]
    Forward(#[from] ForwardError),

    #[error("HTTP Client Error: {0}")]
    HttpClient(String),

    #[error(transparent)]
    Join(#[from] tokio::task::JoinError),

    #[error(transparent)]
    Hyper(#[from] hyper::Error),

    #[error(transparent)]
    Http(#[from] http::Error),

    #[error("Triton error: {0}")]
    TritonError(String),

    #[error("Invalid Triton output: {0}")]
    InvalidTritonOutput(String),
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Failed to read config file: {0}")]
    FileReadError(#[from] std::io::Error),

    #[error("Failed to parse YAML: {0}")]
    YamlParseError(#[from] serde_yaml::Error),

    #[error("Missing required field in policy '{policy}': {field}")]
    MissingPolicyField { policy: String, field: String },

    #[error("Missing required field in LLM '{llm}': {field}")]
    MissingLlmField { llm: String, field: String },
}

#[derive(Debug, thiserror::Error)]
#[error("Forward error: {0}")]
pub struct ForwardError(#[from] Box<dyn std::error::Error + Send + Sync>);

impl ForwardError {
    pub fn new<E>(error: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        ForwardError(Box::new(error))
    }
}

impl From<hyper::Error> for ForwardError {
    fn from(err: hyper::Error) -> Self {
        ForwardError::new(err)
    }
}

impl From<reqwest::Error> for GatewayApiError {
    fn from(error: reqwest::Error) -> Self {
        GatewayApiError::HttpClient(error.to_string())
    }
}

//
// Implement conversion from Infallible to your custom error.
// Since Infallible has no possible values, the match arm is unreachable.
//
impl From<Infallible> for GatewayApiError {
    fn from(err: Infallible) -> Self {
        match err {}
    }
}

#[cfg(test)]
mod tests {
    use serde::de::Error;

    use super::*;

    #[test]
    fn test_json_error() {
        let error = serde_json::Error::custom("invalid json");
        let gateway_api_error = GatewayApiError::Json(error);
        assert_eq!(gateway_api_error.to_string(), "invalid json");
    }

    #[test]
    fn test_io_error() {
        let error = std::io::Error::new(std::io::ErrorKind::Other, "io error");
        let gateway_api_error = GatewayApiError::Io(error);
        assert_eq!(gateway_api_error.to_string(), "io error");
    }

    #[test]
    fn test_yaml_error() {
        let error = serde_yaml::Error::custom("invalid yaml");
        let gateway_api_error = GatewayApiError::Yaml(error);
        assert_eq!(gateway_api_error.to_string(), "invalid yaml");
    }

    #[test]
    fn test_openai_error() {
        let gateway_api_error = GatewayApiError::OpenAI("OpenAI error".to_string());
        assert_eq!(
            gateway_api_error.to_string(),
            "OpenAI API Error: OpenAI error"
        );
    }

    #[test]
    fn test_uri_error() {
        let gateway_api_error = GatewayApiError::Uri("invalid uri".to_string());
        assert_eq!(
            gateway_api_error.to_string(),
            "Could not parse URI: invalid uri"
        );
    }

    #[test]
    fn test_internal_server_error() {
        let gateway_api_error = GatewayApiError::InternalServer;
        assert_eq!(gateway_api_error.to_string(), "Internal Server Error");
    }

    #[test]
    fn test_model_not_found_error() {
        let gateway_api_error = GatewayApiError::ModelNotFound("model not found".to_string());
        assert_eq!(
            gateway_api_error.to_string(),
            "Model not found: model not found"
        );
    }

    #[test]
    fn test_forward_error() {
        let error = Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "forward error",
        ));
        let forward_error = ForwardError(error);
        assert_eq!(forward_error.to_string(), "Forward error: forward error");
    }
}
