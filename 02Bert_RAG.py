#RAG with Bert

from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
import torch

# 初始化分词器和模型
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained(
    "facebook/rag-sequence-nq",
    index_name="exact",
    use_dummy_dataset=True
)
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)

# 输入问题
input_question = "What is the capital of France?"

# 编码输入
inputs = tokenizer(input_question, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# 使用RAG模型进行检索和生成，设置一个适当的最大长度
with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids, 
        attention_mask=attention_mask,
        max_new_tokens=50  # 允许生成的新token数量，根据需要调整
    )

# 解码生成的答案
answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(answer)
