"""# CodeLinker package.

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

The we can define the schema of the return value:
```python
class HelloWorldSchema(BaseModel):
    message: str = Field(description="the message to be returned")
```

Then we can use the `cl` object to wrap the function you want to call:
```python
@cl.smartFunc()
def hello_world() -> HelloWorldSchema:
    '''Say hello to the world'''
```

Now we can call the function and let the model generate the return value:
```python
result = hello_world()
print(result.message)
# sample output:
# Hello, World!
```

The function wrapped by `cl.smartFunc` will have extra key-world arguments that can be used to control the output of the model:
- `messages`: a list of messages that will be inserted into the beginning of the prompt
- `images`: a list of images that will be inserted into the end of the prompt, following openai's message image format
- `reply_format`: a reply format is a instance of `StructureSchema` that helps the model to better understand the context of the conversation.



"""


from .linker import CodeLinker
from .config import CodeLinkerConfig
from .events import EventSink, EventProcessor
from .models import Channels, ChannelTag