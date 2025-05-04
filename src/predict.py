import os
import torch
import preprocessing as preprocessing
import argparse


def predict(directory):
    """This function prints the prediction of the class of each audio file in the specified directory

    Parameters
    ----------
    directory: str, required
        Path to audio files directory
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("../saved_models/CNN_model.pt").to(device)
    label_map = torch.load("../saved_models/Encoder_mapping.pt")

    for file_name in os.listdir(directory):
        x_data = []
        filepath = os.path.join(directory, file_name)
        filtered_audio = preprocessing.filter_and_split(filepath)
        for segment in filtered_audio:
            bispec = preprocessing.create_bispectrum(segment)
            x_data.append(bispec)

        x = torch.stack(x_data).to(device)
        y_hat = model(x)
        prediction = torch.argmax(y_hat.mean(dim=0)).item()

        print(file_name, " Prediction: ", label_map[prediction])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Example script with a string argument"
    )
    parser.add_argument(
        "--d",
        type=str,
        default="inference_audio_files",
        help="Path to audio files directory (default: '../inference_audio_files')",
    )
    args = parser.parse_args()

    predict(args.d)
