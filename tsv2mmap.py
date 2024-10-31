from transformers import BertTokenizerFast
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

def process_triplet(queries,poses,negs):
    ## get query_input_ids
    queries = [q_mark+" "+query for query in queries]
    query_input_ids = tokenizer(queries,padding='max_length',truncation=True,return_tensors='pt',max_length=query_max_len)['input_ids']
    query_input_ids[query_input_ids==tokenizer.pad_token_id] = tokenizer.mask_token_id ## called *query augmentation* in the paper

    ## get_doc_input_ids
    poses = [d_mark+" "+pos for pos in poses]
    pos_input_ids = tokenizer(poses,padding='max_length',truncation=True,return_tensors='pt',max_length=doc_max_len)['input_ids']

    negs = [d_mark+" "+neg for neg in negs]
    neg_input_ids = tokenizer(negs,padding='max_length',truncation=True,return_tensors='pt',max_length=doc_max_len)['input_ids']

    return query_input_ids,pos_input_ids,neg_input_ids

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="google-bert/bert-base-multilingual-uncased")
    parser.add_argument("--query_max_len",type=int,default=32)
    parser.add_argument("--doc_max_len",type=int,default=200)
    parser.add_argument("--triple_path", required=True)
    parser.add_argument("--batch_size",type=int,default=10_000)
    parser.add_argument("--save_dir", default="./mmap/")
    parser.add_argument("--num_samples",type=int,default=21875) #one pos_doc, neg_doc per query
    args = parser.parse_args()

    tokenizer = BertTokenizerFast.from_pretrained(args.base_model)
    query_max_len = args.query_max_len
    doc_max_len = args.doc_max_len
    triplet_path = args.triple_path
    batch_size = args.batch_size
    num_samples = args.num_samples

    q_mark,d_mark = "[Q]","[D]"
    additional_special_tokens = [q_mark,d_mark]
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens":additional_special_tokens,
        }
    )
    os.makedirs(args.save_dir,exist_ok=True)
    query_mmap = np.memmap(os.path.join(args.save_dir,'queries.mmap'), dtype='float32',mode='w+',shape=(num_samples,query_max_len))
    pos_mmap   = np.memmap(os.path.join(args.save_dir,'pos_docs.mmap'),dtype='float32',mode='w+',shape=(num_samples,doc_max_len))
    neg_mmap   = np.memmap(os.path.join(args.save_dir,'neg_docs.mmap'),dtype='float32',mode='w+',shape=(num_samples,doc_max_len))

    total = 0
    progress_bar = tqdm(range(num_samples),desc='processing triplet data...')

    df = pd.read_csv(triplet_path, sep='\t')#revised
    queries,poses,negs = [],[],[]
    for line in df.to_dict('records'):
      query,pos,neg = line['query'], line['pos_doc'], line['neg_doc']
      queries.append(query)
      poses.append(pos)
      negs.append(neg)

      if len(queries) == batch_size:
          query_input_ids,pos_input_ids,neg_input_ids = process_triplet(queries,poses,negs)

          query_mmap[total:total+batch_size] = query_input_ids.numpy().astype(np.int16)
          pos_mmap[  total:total+batch_size] = pos_input_ids.numpy().astype(np.int16)
          neg_mmap[  total:total+batch_size] = neg_input_ids.numpy().astype(np.int16)

          total += batch_size
          progress_bar.update(batch_size)
          queries,poses,negs = [],[],[]

    if len(queries) > 0:
        current_size = len(queries)
        query_input_ids,pos_input_ids,neg_input_ids = process_triplet(queries,poses,negs)

        query_mmap[total:total+current_size] = query_input_ids.numpy().astype(np.float32)
        pos_mmap[total:total+current_size] = pos_input_ids.numpy().astype(np.float32)
        neg_mmap[total:total+current_size] = neg_input_ids.numpy().astype(np.float32)
        progress_bar.update(len(queries))
        assert current_size + total == num_samples

    query_mmap.flush()
    pos_mmap.flush()
    neg_mmap.flush()
