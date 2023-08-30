import gradio as gr
from datasets import Dataset, load_dataset
import pandas as pd

# dataset = Dataset.from_dict({'data': ['no data yet']})

def sample_from_iterable_dataset(dataset, num_samples):
    sampled_data = []
    
    for example in dataset:
        sampled_data.append(example)
        if len(sampled_data) >= num_samples:
            break
    
    return sampled_data



def load_dataset_by_name(dataset_name, dataset_config):
    info = 'Loading...'
    try:
        if dataset_config not in [None, ""]:
            dataset = load_dataset(dataset_name, dataset_config, streaming=True)
            info = "Loaded dataset: " + dataset_name + " with config: " + dataset_config
        else:
            dataset = load_dataset(dataset_name, streaming=True)
            info = "Loaded dataset: " + dataset_name
        num_samples_to_get = 10
        samples = sample_from_iterable_dataset(dataset["train"], num_samples_to_get)
        column_names = list(samples[0].keys())

        data_dict = {column: [example["text"] for example in samples] for column in column_names}
        print(column_names)
        dropdown_text_column.update(choices=column_names)
        df = pd.DataFrame(data_dict)


    except Exception as e:
        info = f'Error loading dataset "{dataset_name}": {str(e)}'
        df = None
    return info, df

with gr.Blocks() as demo:
    gr.Markdown('''
                    # Language model lab
                
                    [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

                    This app allows you train your own language models.
                ''')
    with gr.Tab('Dataset'):
        with gr.Row():
            textbox_dataset_name = gr.Textbox(label='Hugging Face dataset to use as training data', value='oscar', placeholder='e.g. oscar')
            textbox_dataset_config_name = gr.Textbox(label='Dataset config (optional)', value='unshuffled_deduplicated_en', placeholder='e.g. unshuffled_deduplicated_en')
        btn_load_dataset = gr.Button('Load dataset')
        textbox_dataset_status = gr.Textbox(label='Status')
        with gr.Row():
            dropdown_text_column = gr.Dropdown(label="Select text column")
        dataframe_dataset = gr.Dataframe(label = 'Dataset', datatype=["number", "str"])

    with gr.Tab('Preprocessing'):
        gr.Textbox()
    
    with gr.Tab('Training'):
        gr.Textbox(label='Hugging Face model to start from', placeholder='e.g. bert-base-uncased')
    with gr.Tab('Evaluation'):
        gr.Markdown('Flip text or image files using this demo.')
    
    event_load_dataset = btn_load_dataset.click(
                                fn = load_dataset_by_name, 
                                inputs=[textbox_dataset_name, textbox_dataset_config_name], 
                                outputs=[textbox_dataset_status, dataframe_dataset]
                            )



if __name__ == '__main__':
    demo.launch()