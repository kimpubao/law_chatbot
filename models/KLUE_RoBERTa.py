{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "2d344cdc",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/mnt/d/WSL/law_chatbot/.venv/lib/python3.10/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html\n",
      "  from .autonotebook import tqdm as notebook_tqdm\n",
      "/mnt/d/WSL/law_chatbot/.venv/lib/python3.10/site-packages/transformers/utils/hub.py:111: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.\n",
      "  warnings.warn(\n",
      "Some weights of RobertaModel were not initialized from the model checkpoint at klue/roberta-base and are newly initialized: ['pooler.dense.bias', 'pooler.dense.weight']\n",
      "You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "torch.Size([1, 768])\n"
     ]
    }
   ],
   "source": [
    "from transformers import AutoTokenizer, AutoModel\n",
    "import torch\n",
    "\n",
    "model_name = \"klue/roberta-base\"\n",
    "\n",
    "tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
    "model = AutoModel.from_pretrained(model_name)\n",
    "\n",
    "text = \"사기죄는 어떨 때 성립되나요?\"\n",
    "inputs = tokenizer(text, return_tensors=\"pt\")\n",
    "outputs = model(**inputs)\n",
    "\n",
    "# [CLS] 토큰 기준 임베딩\n",
    "embedding = outputs.last_hidden_state[:, 0, :]\n",
    "print(embedding.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4ed047f7",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": ".venv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
