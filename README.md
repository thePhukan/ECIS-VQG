# Entity-centric Information-seeking Question Generation from Videos
Code and Sample Data for the Paper: "Entity-centric Information-seeking Question Generation from Videos"
## Requirements
- conda create -n vqg python=3.8.5
- conda activate vqg
- pip install -r requirement.txt
- pip install openai [_only if you want to run the gpt3_5_summariser code_]


## GitHub Repository Contents:
- **Code Implementations:** The "_bart_Ce.py_" file contains BART implementation with Cross Entropy loss.  The "_bart_CntCe.py_" file contains code for BART fine-tuning with Contrastive + Cross Entropy loss. The "_t5_Ce.py_" file is for fine-tuning the T5-base model with Cross Entropy loss, and the "_t5_CntCe.py_" file contains T5 implementation with Contrastive + Cross Entropy loss. The "_gpt3_5_summariser.py_" file holds the code to generate summaries from the gpt-3.5-turbo model, and the "_bert_classifier.py_" implements our BERT-based classifier. The "__bart_CntCe_Ec.py_" contains the code for BART fine-tuning with Clip embeddings and Contrastive + Cross Entropy loss, while the "__bart_CntCe_Er.py_" file contains the code for BART fine-tuning with ResNext embeddings and Contrastive + Cross Entropy loss. Similarly, the "__t5_CntCe_Ec.py_" file contains T5 implementation with Clip embeddings and Contrastive + Cross Entropy loss, while the "__t5_CntCe_Er.py_" file contains the code for T5 fine-tuning with ResNext embeddings and Contrastive + Cross Entropy loss.
- **Dataset:** We provide a sample of our dataset in the "_sample_data.xlsx_" file. The full dataset (including videos) will be shared upon acceptance of the paper.
