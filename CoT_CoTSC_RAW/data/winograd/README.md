---
dataset_info:
  features:
  - name: text
    dtype: string
  - name: pronoun
    dtype: string
  - name: pronoun_loc
    dtype: int32
  - name: quote
    dtype: string
  - name: quote_loc
    dtype: int32
  - name: options
    sequence: string
  - name: label
    dtype:
      class_label:
        names:
          '0': '0'
          '1': '1'
  - name: source
    dtype: string
  - name: template_name
    dtype: string
  - name: template
    dtype: string
  - name: rendered_input
    dtype: string
  - name: rendered_output
    dtype: string
  splits:
  - name: test
    num_bytes: 1265512.0
    num_examples: 2184
  download_size: 183645
  dataset_size: 1265512.0
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
---
