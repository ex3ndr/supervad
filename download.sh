set -e 

BUCKET_URL="https://datasets.korshakov.com"
if [ -n "$LOCAL_DATASETS" ]; then
    BUCKET_URL="$LOCAL_DATASETS"
fi

FILES=(
    "supervad-1/vad_test"
    "supervad-1/vad_train"
)

for FILE in ${FILES[@]}
do

    # Download
    mkdir -p ./datasets/$FILE
    aria2c -x8 -s 16 --file-allocation=none $BUCKET_URL/$FILE.tar.gz -o ./datasets/$FILE/archive.tar.gz

    # # Unpack
    tar -xvf "./datasets/$FILE/archive.tar.gz" -C "./datasets/$FILE" --strip-components=1

    # # Remove
    rm "./datasets/$FILE/archive.tar.gz"
done
