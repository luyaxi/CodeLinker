# Introduction

The configuration object should be instantiated from the `CodeLinkerConfig` class.
The configuration object is mainly used to store the API keys and model aliases that will be used during the execution of the package.
We recommend to use toml file to store the configuration, and use the `from_toml` method to load the configuration from the file.

## Sample Configuration

```toml
max_retry_times = 3

[[api_keys.gpt-4o]]
api_key = "sk-1234"
model = "gpt-4o"

[[api_keys.gpt-4o-mini]]
api_key = "sk-5678"
model = "gpt-4o-mini"

[request]
format = "tool_call" # how to obtain structured data from the model, can be chat, tool_call, or function_call
json_mode = false # whether to use json mode to send the request
dynamic_json_fix = true # Trying to fix the errored json if the json_mode is enabled
default_completions_model = "gpt-4o"
default_timeout = 60
default_request_lib = "openai" # which request library to use, currently only support openai
use_cache = true # whether read the saved completions from the cache, will accelerate the execution with same inputs
save_completions = false
save_complections_path = ".cache/completions"
```

To load the configuration from the file (e.g. `config.toml`), you can use the following code:

```python
config = CodeLinkerConfig.from_toml("config.toml")
```
