## make large tsv file to memory-mapping file for efficient pytorch indexing
python tsv2mmap.py \
    --triple_path /content/drive/MyDrive/DIS/train_triple.tsv\ #TODO
    #--doc_max_len 512
    #--query_max_len 200
  
## train
python train_colbert.py

## using trained model to produce document embedding
accelerate launch doc2embedding.py \
    --pretrained_model_path wandb/latest-run/files/step-4000 \ #TODO
    --output_dir embedding/colbert \
    --collection_path corpus.json #TODO

## build faiss index for efficient retrieval
python build_index.py --embedding_dir embedding/colbert --output_path embedding/colbert/ivfpq.faiss.index

## conduct retrieval
#python retrieve.py \
#    --embedding_dir embedding/colbert \
#    --faiss_index_path embedding/colbert/ivfpq.faiss.index \
#    --pretrained_model_path wandb/latest-run/files/step-400000 \
#    --query_path data/queries.dev.small.tsv \
#    --output_path ranking.tsv

## calculate score
#python score.py --qrel_path data/qrels.dev.small.tsv --ranking_path ranking.tsv
