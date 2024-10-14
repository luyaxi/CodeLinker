# Structured Generation

Currently, the package support two types of structured generation with language models: `Function Definition` and `Return Declearation`.

## Function Definition

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

### Extra Arguments

The function wrapped by `cl.smartFunc` will have extra key-world arguments that can be used to control the output of the model:

- `messages`: a list of messages that will be inserted into the beginning of the prompt
- `images`: a list of images that will be inserted into the end of the prompt, following openai's message image format
- `reasoning_format`: a reasoning format is a instance of `StructureSchema` that helps the model to better understand the context of the conversation with Chain-of-Thought techs.

Other key-world arguments will be passed to the request library to control the request behavior.

## Return Declearation

Other wise, if you do not want to define the function, you can use the `exec` method to call the function:

```python
result = await cl.exec(return_type=HelloWorldSchema)
```

To be noticed, the `exec` method is an async method, so you need to use `await` to call it.

The `exec` method also support the following key-world arguments:

- `prompt`: a string that will be used as the prompt to the model.
- `messages`: a list of messages that will be inserted into the beginning of the prompt.
- `model`: the specific model that will be used to generate the return value.
- `reasoning_format`: a reasoning format is a instance of `StructureSchema` that helps the model to better understand the context of the conversation with Chain-of-Thought techs.
- `schema_validation`: whether to validate the return value with the schema defined in the `return_type` argument.
- `images`: a list of images that will be inserted into the end of the prompt, following openai's message image format.
