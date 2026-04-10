#!/usr/bin/env bash
set -eu

out_dir="${1:-data/mnist}"

base_urls="
https://ossci-datasets.s3.amazonaws.com/mnist
http://yann.lecun.com/exdb/mnist
"

files="
train-images-idx3-ubyte
train-labels-idx1-ubyte
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte
"

mkdir -p "$out_dir"

download_file() {
    file="$1"
    gz_name="${file}.gz"
    gz_path="${out_dir}/${gz_name}"
    raw_path="${out_dir}/${file}"

    if [ -f "$raw_path" ]; then
        printf 'MNIST file already present: %s\n' "$raw_path"
        return 0
    fi

    for base_url in $base_urls; do
        url="${base_url}/${gz_name}"
        printf 'Downloading %s\n' "$url"
        if curl --fail --location --silent --show-error "$url" -o "$gz_path"; then
            gzip -dc "$gz_path" > "$raw_path"
            rm -f "$gz_path"
            printf 'Saved %s\n' "$raw_path"
            return 0
        fi
    done

    rm -f "$gz_path"
    printf 'Failed to download %s from all configured sources\n' "$gz_name" >&2
    return 1
}

for file in $files; do
    download_file "$file"
done

printf 'MNIST dataset is ready in %s\n' "$out_dir"
