from top2vec import Top2Vec # For loading the Top2Vec model

# Load the Top2Vec model we created in create_model.py
model = Top2Vec.load('top2vec_model')

#get top 10 topics
topic_words, word_scores, topic_nums =  model.get_topics(10)
for topic in topic_words:
    print(topic)
    print()

#search for documents related to topic 2 [0,1,(2)], the third element
documents, document_scores, document_ids = model.search_documents_by_topic(topic_num=2, num_docs=100)  
for doc, score, doc_id in zip(documents, document_scores, document_ids):                              
    print(f"Document: {doc_id}, Score: {score}")
    print("-----------")
    print(doc)
    print("-----------")
    print()