import gradio as gr
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModel, DataCollatorForLanguageModeling, RobertaConfig, RobertaForMaskedLM, TrainingArguments, Trainer
from datetime import datetime
import sys

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def isatty(self):
        return False    

sys.stdout = Logger("output.log")

def read_logs():
    sys.stdout.flush()
    with open("output.log", "r") as f:
        return f.read()

# dataset = Dataset.from_dict({'data': ['no data yet']})

# column_names = []
df_dataset = []
df_text = []


def update_dataset_columns(df_sample):
    print("change event")
    column_names = list(df_sample)
    return gr.Dropdown(choices=column_names, interactive=True)

def sample_from_iterable_dataset(dataset, num_samples):
    sampled_data = []
    
    for example in dataset:
        sampled_data.append(example)
        if len(sampled_data) >= num_samples:
            break
    print(sampled_data)
    return sampled_data



# def load_dataset_by_name(dataset_name, dataset_config):
def load_dataset_by_name(dataset_name):
    print("Load dataset by name")
    try:
        global df_dataset
        # if dataset_config not in [None, ""]:
        #     dataset = load_dataset(dataset_name, dataset_config, streaming=True)
        # else:
        #     dataset = load_dataset(dataset_name, streaming=True)

        # dataset = load_dataset("parquet", data_files=dataset_name, streaming=True)
        df_dataset = pd.read_parquet(dataset_name)
        # num_samples_to_get = 10
        # samples = sample_from_iterable_dataset(dataset["train"], num_samples_to_get)
        # dropdown_text_column(choices=column_names)
        # df = pd.DataFrame(samples)
        return df_dataset.sample(5), df_dataset.sample(5)

    except Exception as e:
        return None



def pretrain(column_name, hf_model, max_length, progress=gr.Progress(track_tqdm=True)):
    print("column_name", column_name)
    print("hf_model", hf_model)
    print("max_length", max_length)
    ds_dataset = Dataset.from_pandas(df_dataset, preserve_index=False)
    ds_dataset = ds_dataset.select_columns(column_name)
    ds_dataset = ds_dataset.rename_column(column_name, "labels")
    shuffled_dataset = ds_dataset.shuffle(seed=42)
    shuffled_dataset = shuffled_dataset.train_test_split(train_size=0.7)
    tokenizer = AutoTokenizer.from_pretrained(hf_model, use_fast=True)

    def encode(batch):
        return tokenizer(batch['labels'], padding=True, truncation=True, max_length=max_length,return_tensors="pt")
    
    shuffled_dataset["train"].set_transform(encode)
    shuffled_dataset["test"].set_transform(encode)

    model = AutoModel.from_pretrained(hf_model)

    # Set a configuration for our RoBERTa model
    config = RobertaConfig(
        vocab_size=50265,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )
    # Initialize the model from a configuration without pretrained weights
    model = RobertaForMaskedLM(config=config)
    print('Num parameters: ',model.num_parameters())

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15,
    )

    training_args = TrainingArguments(
        output_dir='./models',
        overwrite_output_dir=True,
        evaluation_strategy = 'steps',
        num_train_epochs=1000,
        learning_rate=1e-5,
        weight_decay=0.01,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        save_steps=1024,
        eval_steps=1024,
        save_total_limit=3,
        ignore_data_skip=True,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        fp16=True
    )
    # Create the trainer for our model
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=shuffled_dataset["train"],
        eval_dataset=shuffled_dataset["test"],
        #prediction_loss_only=True,
    )

    trainer.train()
    now = datetime.now()
    str_datetime = now.strftime("%dd/%mm/%YYYY")
    trainer.save_model('./models/' + hf_model + "_" + str_datetime)
    return "done"

def sample_data():
    return df_dataset.sample(5)

with gr.Blocks() as demo:
    gr.Markdown('''
                    # Language model lab
                
                    [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

                    This app allows you train your own language models.
                ''')
    with gr.Tab('Dataset'):
        with gr.Row():
            textbox_dataset_name = gr.Textbox(label='Parquet dataset to use as training data', placeholder='./data/raw/example.pq')
            # textbox_dataset_name = gr.Textbox(label='Hugging Face dataset to use as training data', value='oscar', placeholder='e.g. oscar')
            # textbox_dataset_config_name = gr.Textbox(label='Dataset config (optional)', value='unshuffled_deduplicated_en', placeholder='e.g. unshuffled_deduplicated_en')
        btn_load_dataset = gr.Button('Load dataset')
        dataframe_dataset = gr.Dataframe(label='Dataset', interactive=False, wrap=True)
        btn_sample_dataset = gr.Button("Sample data")
        btn_sample_dataset.click(sample_data, outputs=[dataframe_dataset])
        




    # with gr.Tab('Preprocessing'):
    #     gr.Textbox()
    
    with gr.Tab('Training'):
        dataframe_processed_dataset = gr.Dataframe(label = 'Processed dataset', interactive=False, wrap=True)
        btn_sample_dataset = gr.Button("Sample data")
        dropdown_text_column = gr.Dropdown(label="Select column containing text to train on", allow_custom_value=True, interactive=True)
        btn_sample_dataset.click(sample_data, outputs=[dataframe_processed_dataset])

        textbox_model_name = gr.Textbox(label='Hugging Face model to start from', placeholder='e.g. bert-base-uncased')
        number_model_max_length = gr.Number(label="Model max token length", value=512)
        btn_pretrain_model = gr.Button('Pretrain model')
        textbox_pretrain_status = gr.Textbox(label="Status")
        textbox_logs = gr.Textbox(label="Logs", lines=5, max_lines=5)
        
    with gr.Tab('Evaluation'):
        gr.Markdown('Flip text or image files using this demo.')

    btn_load_dataset.click(load_dataset_by_name, inputs=[textbox_dataset_name], outputs=[dataframe_dataset, dataframe_processed_dataset])
    dataframe_dataset.change(update_dataset_columns, inputs=[dataframe_dataset], outputs=[dropdown_text_column])
    btn_pretrain_model.click(pretrain, inputs=[dropdown_text_column, textbox_model_name, number_model_max_length], outputs=[textbox_pretrain_status])
    demo.load(read_logs, None, textbox_logs, every=1)



if __name__ == '__main__':
    demo.launch()