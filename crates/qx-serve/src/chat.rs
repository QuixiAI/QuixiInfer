//! Chat prompts: the model's own chat template, rendered with minijinja.
use anyhow::{anyhow, Result};
use minijinja::{Environment, Error, ErrorKind};

pub struct Template {
    env: Environment<'static>,
}

impl Template {
    pub fn new(source: String) -> Result<Self> {
        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |msg: String| -> Result<String, Error> { Err(Error::new(ErrorKind::InvalidOperation, msg)) });
        env.add_template_owned("chat", source).map_err(|e| anyhow!("chat template: {e}"))?;
        Ok(Self { env })
    }

    /// Renders `messages` (OpenAI-style JSON objects) with a generation prompt. `thinking` false asks the model to
    /// answer without a reasoning block.
    pub fn render(&self, messages: &serde_json::Value, tools: Option<&serde_json::Value>, thinking: bool) -> Result<String> {
        let t = self.env.get_template("chat").unwrap();
        let ctx = minijinja::context! {
            messages => minijinja::Value::from(minijinja::value::Serde(messages)),
            tools => tools.map(|t| minijinja::Value::from(minijinja::value::Serde(t))),
            add_generation_prompt => true,
            enable_thinking => thinking,
        };
        t.render(ctx).map_err(|e| anyhow!("chat template: {e:#}"))
    }
}
