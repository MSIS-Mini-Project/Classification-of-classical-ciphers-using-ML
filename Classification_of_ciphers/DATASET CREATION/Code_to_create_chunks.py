import os


book_dir = 'OPEN_SOURCE_TEXTBOOK'
output_dirs = ['PT_columnar', 'PT_dcolumnar', 'PT_feistal', 'PT_vernmam',"PT_vigenere"]
max_chars = 1000
num_files_per_dir = 18000


for directory in output_dirs:
    os.makedirs(directory, exist_ok=True)

def split_text(content, max_chars):

    segments = []
    start = 0
    while start < len(content):
        end = min(start + max_chars, len(content))
        if end < len(content) and content[end].isalnum():
            end = content.rfind(' ', start, end) + 1
        segments.append(content[start:end].strip())
        start = end
    return segments

file_count = {dir_name: 0 for dir_name in output_dirs}
output_dir_index = 0


for book_file in sorted(os.listdir(book_dir)):
    book_path = os.path.join(book_dir, book_file)
    try:
        with open(book_path, 'r', encoding='utf-8', errors='ignore') as infile:
            content = infile.read()


        segments = split_text(content, max_chars)

        for segment in segments:
            current_dir = output_dirs[output_dir_index]

            # Stop if max files reached per directory
            if file_count[current_dir] >= num_files_per_dir:
                output_dir_index += 1
                if output_dir_index >= len(output_dirs):
                    print("Max files reached for all directories.")
                    break
                current_dir = output_dirs[output_dir_index]

            output_file = os.path.join(current_dir, f"{file_count[current_dir] + 1}.txt")
            with open(output_file, 'w', encoding='utf-8') as outfile:
                outfile.write(segment)


            file_count[current_dir] += 1

    except UnicodeDecodeError:
        print(f"Skipping file {book_file} due to encoding issues.")

    if output_dir_index >= len(output_dirs):
        break  # All directories reached limit

print("Text successfully split across directories.")
