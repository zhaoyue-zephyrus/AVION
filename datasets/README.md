# Preparing datasets

Please download the (selected) datasets from the official websites and place or sim-link them under `<YOUR_AVION_PATH>/datasets/`.

```bash
<YOUR_AVION_PATH>/datasets/
    EK100/
    Ego4D/
    Kinetics/
```


## Ego4D

We mostly follow the pre-processing steps in [LAVILA](https://github.com/facebookresearch/LaViLa/blob/main/datasets/README.md).

1. Download [Ego4D videos](https://ego4d-data.org/docs/start-here/#download-data) (license is required).

2. Preprocess: We cut each video into 15-second-long chunks (without overlap) and resize the smaller size to 288 pixels for faster IO. 

* Note that the chunk length is significantly smaller than in the original LAVILA (5 minutes). You can customize it based on your own hardware and training throughput requirements (See more details in our paper). Remember to change `video_chunk_length` accordingly. 

3. Download annotations

    a. Download [video-narrative pairs for train split](https://dl.fbaipublicfiles.com/lavila/metadata/ego4d/ego4d_train.pkl) and [val split](https://dl.fbaipublicfiles.com/lavila/metadata/ego4d/ego4d_val.pkl) to `<YOUR_AVION_PATH>/datasets/Ego4D` (if you want to train LAVILA from scratch).

    b. Download the [rephrased video-narrative pairs](https://dl.fbaipublicfiles.com/lavila/metadata/ego4d/ego4d_train.rephraser.no_punkt_top3.pkl) and [pseudo-narrated pairs](https://dl.fbaipublicfiles.com/lavila/metadata/ego4d/ego4d_train.narrator_63690737.return_10.pkl) generated by the Rephraser and Narrator in LAVILA.

    c. We also provide a [UTBox link](https://utexas.box.com/s/assad9b6pg1opsmwvrp86o0mpdgtp1uh).

The fold should look like this:
```bash
<YOUR_AVION_PATH>/datasets/
    Ego4D/
        ego4d_train.pkl
        ego4d_train.narrator_63690737.return_10.pkl
        ego4d_train.rephraser.no_punkt_top3.pkl
        ego4d_val.pkl
        video_320px_15sec/
            000786a7-3f9d-4fe6-bfb3-045b368f7d44.mp4/
                0.mp4
                15.mp4
                30.mp4
            000a3525-6c98-4650-aaab-be7d2c7b9402.mp4/
                0.mp4
            ...
```


## EPIC-Kitchens-100 (EK-100)

1. Download videos.

    a. For raw videos, please download them from [https://epic-kitchens.github.io/](https://epic-kitchens.github.io/).

    b. The raw videos are huge (~1 TB). As an alternative, please check out a [resized version](https://utexas.box.com/s/l7ij81ie5q07p9fdg0vtejihq61liln9).

2. Preprocess

    Similar to Ego4D, we also cut videos into 15-second-long chunks and resize their smaller size. We provide a [resized, chunked version](https://utexas.box.com/v/ek100-320p-15sec-30fps).

    ```bash
    wget https://utexas.box.com/shared/static/rulikvams7leevaej74bn6oyo6gviybs.zip -O EK100_320p_15sec_30fps_libx264.zip
    ```

3. Download annotations

    ```bash
    # Assume that you are under `datasets/EK100/`
    git clone https://github.com/epic-kitchens/epic-kitchens-100-annotations
    ```

4. (For EK-100 MIR)

    a. Generate the relevancy matrix of train/val splits using [the official code](https://github.com/mwray/Joint-Part-of-Speech-Embeddings).

    b. (Recommended) The generated result has some randomness. Therefore, we also provide the [replica of train split](https://dl.fbaipublicfiles.com/lavila/metadata/EK100/caption_relevancy_EPIC_100_retrieval_train.pkl) and [val split](https://dl.fbaipublicfiles.com/lavila/metadata/EK100/caption_relevancy_EPIC_100_retrieval_test.pkl) we used to avoid fluctuation. Please put them in the folder `<YOUR_AVION_PATH>/datasets/EK100/epic-kitchens-100-annotations/retrieval_annotations/relevancy/`.


The folder should look like this:
```bash
<YOUR_AVION_PATH>/datasets/
    EK100/
        epic-kitchens-100-annotations/
            EPIC_100_train.csv
            EPIC_100_validation.csv
            ...
            retrieval_annotations/relevancy/  # This appears if you do 4.
                caption_relevancy_EPIC_100_retrieval_train.pkl
                caption_relevancy_EPIC_100_retrieval_test.pkl
        video_320p_15sec/
            P01/
                P01_01.MP4/
                    0.MP4
                    15.MP4
                    ...
                    1650.MP4
                P01_02.MP4
                ...
                P01_19.MP4
            P02/
                P02_01.MP4
                P02_02.MP4
                ...
                P02_15.MP4
            ...
```


## Kinetics

1. Download [kinetics-400 videos](https://github.com/cvdfoundation/kinetics-dataset).

2. (Optional) You can resize the smaller size to be a fixed value (e.g. 320 pixels). This can reduce the storage space by more than 60%.

3. Prepare the annotation list in the format below (following [VideoMAE](https://github.com/MCG-NJU/VideoMAE/blob/main/DATASET.md)):

    ```
    dataset_root/video_1.mp4 num_frames_1 label_1
    dataset_root/video_2.mp4 num_frames_2 label_2
    dataset_root/video_3.mp4 num_frames_3 label_3
    ...
    dataset_root/video_N.mp4 num_frames_N label_N
    ```

    We provide an annotation list [here](https://utexas.box.com/s/md4ujy6zjsaji4ug4pjbkde5rd2pkqdf).

The folder should look like this:

```bash
<YOUR_AVION_PATH>/datasets/
    Kinetics/
        annotations/
            k400_320p_train_list.txt
            k400_320p_val_list.txt
        train_320px/
            0074cdXclLU_000047_000057.mp4
            0083lvylGBs_000165_000175.mp4
            ...
            zV887gbNi4Q_000139_000149.mp4
        val_320px/
            00cwEcZZcu4_000003_000013.mp4
            00U9x58IraQ_000007_000017.mp4
```
