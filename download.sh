set -e 

BUCKET_URL="https://datasets.korshakov.com"

FILES=(
    "supervad-1/vad_test"
    "supervad-1/vad_train"
)

for FILE in ${FILES[@]}
do

    # Download
    mkdir -p ./datasets/$FILE
    wget $BUCKET_URL/$FILE.tar.gz -O ./datasets/$FILE/archive.tar.gz

    # Unpack
    tar -xvf "./datasets/$FILE/archive.tar.gz" -C "./datasets/$FILE" --strip-components=1

    # Remove
    rm "./datasets/$FILE/archive.tar.gz"
done
