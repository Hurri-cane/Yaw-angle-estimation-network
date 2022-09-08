
# Install
1. Clone the project

    ```Shell
    git clone https://github.com/Hurri-cane/Yaw-angle-estimation-network
    cd Yaw-angle-estimation-network
    ```

2. Create a conda virtual environment and activate it

    ```Shell
    conda create -n YAEN python=3.8 -y
    conda activate YAEN
    ```

3. Install dependencies

    ```Shell
    # If you dont have pytorch
    pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
    
    pip install -r requirements.txt
    ```
    
4. Data preparation

     The directory arrangement of "Yaw angle dataset" should look like:
    ```
    $YAENROOT
    |──graph_weight
    |──train
    ||───heading.data
    ||───heading.png
    |──val
    ||───heading.data
    ||───heading.png
    |──org
    ||───Yaw_angle_dataset.data
    ||───Yaw_angle_dataset.png
    |──readme.md
    ```