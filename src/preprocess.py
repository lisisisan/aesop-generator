import re
import os

input_file = "data/aesop/data.txt"
output_file = "data/aesop/data_processed.txt"

with open(input_file) as f:
    text = f.read().lower()

# Разделяем басни (по двойным пустым строкам) и вставляем уникальный разделитель
stories = re.split(r'\n\s*\n', text)
text_processed = '|||||||||||||||||||'.join(stories)

os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w") as f:
    f.write(text_processed)

print(f"Processed text saved to {output_file}")
