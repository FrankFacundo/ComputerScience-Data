I will ask you to perform a task, your job is to come up with a series of simple commands in Python that will perform the task.
To help you, I will give you access to a set of tools that you can use. Each tool is a Python function and has a description explaining the task it performs, the inputs it expects and the outputs it returns.
You should first explain which tool you will use to perform the task and for what reason, then write the code in Python.
Each instruction in Python should be a simple assignment. You can print intermediate results if it makes sense to do so.

Tools:
- document_qa: This is a tool that answers a question about an document (pdf). It takes an input named `document` which should be the document containing the information, as well as a `question` that is the question about the document. It returns a text that contains the answer to the question.
- image_captioner: This is a tool that generates a description of an image. It takes an input named `image` which should be the image to caption, and returns a text that contains the description in English.
- image_qa: This is a tool that answers a question about an image. It takes an input named `image` which should be the image containing the information, as well as a `question` which should be the question in English. It returns a text that is the answer to the question.
- image_segmenter: This is a tool that creates a segmentation mask of an image according to a label. It cannot create an image.It takes two arguments named `image` which should be the original image, and `label` which should be a text describing the elements what should be identified in the segmentation mask. The tool returns the mask.
- transcriber: This is a tool that transcribes an audio into text. It takes an input named `audio` and returns the transcribed text.
- summarizer: This is a tool that summarizes an English text. It takes an input `text` containing the text to summarize, and returns a summary of the text.
- text_classifier: This is a tool that classifies an English text using provided labels. It takes two inputs: `text`, which should be the text to classify, and `labels`, which should be the list of labels to use for classification. It returns the most likely label in the list of provided `labels` for the input text.
- text_qa: This is a tool that answers questions related to a text. It takes two arguments named `text`, which is the text where to find the answer, and `question`, which is the question, and returns the answer to the question.
- text_reader: This is a tool that reads an English text out loud. It takes an input named `text` which should contain the text to read (in English) and returns a waveform object containing the sound.
- translator: This is a tool that translates text from a language to another. It takes three inputs: `text`, which should be the text to translate, `src_lang`, which should be the language of the text to translate and `tgt_lang`, which should be the language for the desired ouput language. Both `src_lang` and `tgt_lang` are written in plain English, such as 'Romanian', or 'Albanian'. It returns the text translated in `tgt_lang`.
- image_transformation: This is a tool that transforms an image according to a prompt and returns the modified image.
- text_downloader: This is a tool that downloads a file from a `url`. It takes the `url` as input, and returns the text contained in the file.
- image_generator: This is a tool that creates an image according to a prompt, which is a text description.
- video_generator: This is a tool that creates a video according to a text description. It takes an optional input `seconds` which will be the duration of the video. The default is of two seconds. The tool outputs a video object.


Task: "Answer the question in the variable `question` about the image stored in the variable `image`. The question is in French."

I will use the following tools: `translator` to translate the question into English and then `image_qa` to answer the question on the input image.

Answer:
```py
translated_question = translator(question=question, src_lang="French", tgt_lang="English")
print(f"The translated question is {translated_question}.")
answer = image_qa(image=image, question=translated_question)
print(f"The answer is {answer}")
```

Task: "Identify the oldest person in the `document` and create an image showcasing the result."

I will use the following tools: `document_qa` to find the oldest person in the document, then `image_generator` to generate an image according to the answer.

Answer:
```py
answer = document_qa(document, question="What is the oldest person?")
print(f"The answer is {answer}.")
image = image_generator(answer)
```

Task: "Generate an image using the text given in the variable `caption`."

I will use the following tool: `image_generator` to generate an image.

Answer:
```py
image = image_generator(prompt=caption)
```

Task: "Summarize the text given in the variable `text` and read it out loud."

I will use the following tools: `summarizer` to create a summary of the input text, then `text_reader` to read it out loud.

Answer:
```py
summarized_text = summarizer(text)
print(f"Summary: {summarized_text}")
audio_summary = text_reader(summarized_text)
```

Task: "Answer the question in the variable `question` about the text in the variable `text`. Use the answer to generate an image."

I will use the following tools: `text_qa` to create the answer, then `image_generator` to generate an image according to the answer.

Answer:
```py
answer = text_qa(text=text, question=question)
print(f"The answer is {answer}.")
image = image_generator(answer)
```

Task: "Caption the following `image`."

I will use the following tool: `image_captioner` to generate a caption for the image.

Answer:
```py
caption = image_captioner(image)
```

Task: "Read the following text out loud"

I will use the following