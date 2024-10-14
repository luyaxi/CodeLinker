# CodeLinker : Link your code with Language Models

CodeLinker aims to provide functions to link your code with language models.
It builds on top of the Pydatic library and Tool Calling abilities introduced by [OpenAI](https://platform.openai.com/docs/guides/function-calling), which enabling models to generate content according to [Json Schema](https://json-schema.org/).

## Usage

The core concept of this package is to treat language models as a function handler.
By defining a schema for return value of the function, we can call the function and let the model generate the return value.

To start with, we need to first define the configuration that will be used during exection:

```python
config = CodeLinkerConfig(api_keys={
    "gpt-3.5-turbo-16k":[{
        "api_key": "your api key here",
        "model": "model name alias here"
    }]
})
cl = CodeLinker(config)
```

Learn more about the configuration in the [Configuration](docs/configuration.md) page.

## Structured Generation

Currently, the package support two types of structured generation with language models: `Function Definition` and `Return Declearation`.

#### Function Definition

You can define the schema of the return value with pydantic package:

```python
class HelloWorldSchema(BaseModel):
    message: str = Field(description="the message to be returned")
```

Then we can use the `cl` object to wrap the function you want to call:

```python
@cl.smartFunc()
def hello_world() -> HelloWorldSchema:
    """Say hello to the world"""
```

The function's docstring will be passed to models as instruction about what this function should do.
Now we can call the function and let the model generate the return value:

```python
result = hello_world()
print(result.message)
# sample output:
# Hello, World!
```

#### Return Declearation

Other wise, if you do not want to define the function, you can use the `exec` method to call the function:

```python
result = await cl.exec(return_type=HelloWorldSchema)
```

To be noticed, the `exec` method is an async method, so you need to use `await` to call it.

More details about structured generation can be found in the [Structured Generation](docs/structuredgen.md) page.
