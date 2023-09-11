import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os


paragraph = '''
Title: Demonetization: A Critical Analysis of India's Bold Economic Move

Introduction

On November 8, 2016, India embarked on a historic economic experiment by demonetizing its high-denomination currency notes of Rs. 500 and Rs. 1,000. This move, known as demonetization, aimed to combat black money, corruption, counterfeit currency, and promote digital transactions. It sent shockwaves through the nation, sparking debates and discussions about its implications, successes, and failures. This essay critically analyzes demonetization, examining its objectives, outcomes, and its impact on India's economy and society.

Objectives of Demonetization

1. Tackling Black Money:
One of the primary objectives of demonetization was to curb the black money menace. Black money refers to income on which taxes have not been paid. By invalidating high-denomination currency notes, the government aimed to bring undisclosed wealth into the formal economy and tax it appropriately. This, in turn, was expected to increase government revenue and reduce income inequality.

2. Curbing Corruption:
Another goal was to reduce corruption by making it difficult for individuals to hoard large sums of unaccounted cash. Demonetization was expected to disrupt corrupt practices, particularly in real estate, politics, and public services, where large amounts of black money were often exchanged.

3. Promoting Digital Transactions:
The government also sought to encourage digital transactions as part of its broader vision of a cashless economy. By removing high-denomination cash, people were pushed to use digital payment methods, thereby reducing the reliance on physical currency.

4. Countering Counterfeit Currency:
The circulation of counterfeit currency was a significant concern for the Indian economy. Demonetization aimed to eliminate counterfeit notes by introducing new, more secure currency notes.

Outcomes of Demonetization

1. Short-Term Disruption:
The immediate aftermath of demonetization saw significant disruption. Long lines formed at banks and ATMs as people rushed to exchange their old notes for new ones. This resulted in inconvenience, particularly for those in rural areas with limited access to banking facilities.

2. Impact on Black Money:
While demonetization did lead to the identification and disclosure of some black money, its overall impact in this regard remains debatable. Critics argue that most black money is stored in forms other than cash, such as real estate, gold, and foreign assets. Therefore, demonetization alone may not have been sufficient to unearth substantial illicit wealth.

3. Cashless Transactions:
Demonetization did push people towards digital transactions. Mobile wallets, online banking, and digital payment platforms witnessed a surge in usage during and after demonetization. This shift towards a cashless economy had both positive and negative consequences, as discussed later.

4. Economic Impact:
The Indian economy experienced a slowdown in the immediate aftermath of demonetization, as businesses, particularly small and medium-sized enterprises (SMEs), struggled due to a cash crunch. Critics argue that the economic costs outweighed the benefits of demonetization.

Impact on Society

1. Financial Inclusion:
One of the positive outcomes of demonetization was an increased focus on financial inclusion. The government aimed to bring more people into the formal banking system, particularly in rural areas. Programs like Jan Dhan Yojana, which aimed to provide every household with a bank account, gained momentum.

2. Job Losses:
The economic disruption caused by demonetization led to job losses in various sectors, including agriculture, construction, and small-scale industries. The informal labor sector was hit hard, affecting vulnerable populations.

3. Political Ramifications:
Demonetization had significant political implications. While some lauded the move as a bold step against corruption, others criticized it as a political stunt. Its impact on the 2017 Uttar Pradesh state elections and subsequent state elections is a testament to its political significance.

4. Effect on Informal Economy:
India has a large informal economy, and demonetization had a severe impact on this sector. Small businesses and street vendors, which rely heavily on cash transactions, faced challenges in recovering from the sudden cash crunch.

Conclusion

Demonetization was a bold and ambitious economic move by the Indian government with multiple objectives. It aimed to tackle black money, corruption, counterfeit currency, and promote digital transactions. However, the outcomes and impact of demon
'''

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
    )
chunks = text_splitter.split_text(text=paragraph)


def main():

    store_name = "demonetisation"
    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl", "rb") as f:
            # Embeddings loaded from disk.
            VectorStore = pickle.load(f)
    else:
        embeddings = OpenAIEmbeddings(openai_api_key = "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")
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(VectorStore, f)
    
    prompt = "What are outcomes of demonetisation?"

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(openai_api_key = "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", model_name='gpt-3.5-turbo'),
        chain_type='stuff',
        retriever=VectorStore.as_retriever()
    )
    result = qa.run({'query' : prompt})
    print(result)

    result = qa.run({'query' : "Elaborate"})
    print(result)
 
if __name__ == '__main__':
    main()