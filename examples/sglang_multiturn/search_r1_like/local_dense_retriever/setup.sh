#!/bin/bash

echo "1. Creating conda environment..."
conda create -n retriever python=3.10 -y
conda activate retriever
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install datasets faiss-cpu numpy transformers fastapi uvicorn pydantic tqdm huggingface_hub

echo "2. Downloading retrieval data..."
RETRIEVER_DATA_PATH=examples/sglang_multiturn/search_r1_like/local_dense_retriever/retrieve_data
python examples/sglang_multiturn/search_r1_like/local_dense_retriever/download.py --save_path $RETRIEVER_DATA_PATH

echo "3. Merging index files and unzipping corpus file..."
cat $RETRIEVER_DATA_PATH/part_* > $RETRIEVER_DATA_PATH/e5_Flat.index
rm $RETRIEVER_DATA_PATH/part_*
gzip -d $RETRIEVER_DATA_PATH/wiki-18.jsonl.gz

echo "4. Downloading train and test data..."
python examples/data_preprocess/qa_search_test_merge.py --local_dir data/search-r1 --data_sources nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
conda activate verl
python examples/data_preprocess/preprocess_search_r1_dataset.py --local_dir data/search-r1