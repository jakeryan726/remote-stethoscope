import torch
import pandas as pd


def generate_samples(model, df, generate_until, device, scale_factor=1.0):
    """Adds generated data to each class label until the specified threshold

    Parameters
    ----------
    model: VAE, required
        The model to generate the new data using
    df: DataFrame, required
        The dataframe to fill with generated data, must have columns [labels, images]
    generate_until: int, required
        The number of records to fill each class until
    device: str, required
        The device to send the data to before input into the model
    scale_factor: float, optional
        Additional factor to scale the mean of the generated distribution

    Returns
    -------
    DataFrame
        A pandas dataframe with two columns, images and labels, each class in labels having
        at least generated_until records
    """

    generations = []
    model.eval()
    classes = df["labels"].value_counts()
    with torch.no_grad():
        for label, count in classes.items():
            for _ in range(generate_until - count):
                image = (
                    df[df["labels"] == label]
                    .sample(n=1)["images"]
                    .iloc[0]
                    .unsqueeze(0)
                    .to(device)
                )
                mu, log_var = model.encode(image)
                sigma = torch.exp(0.5 * log_var)
                eps = scale_factor * torch.rand_like(mu)
                z = mu + eps * sigma
                generated_image = model.decode(z).squeeze(0).cpu()
                generations.append({"images": generated_image, "labels": label})
    return pd.concat([pd.DataFrame(generations), df], ignore_index=True)


if __name__ == "__main__":
    model = torch.load("../saved models/VAE_model.pt")
    data = torch.load("../bispectrum_train_df.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_with_generations = generate_samples(model, data, 800, device)
    torch.save(data_with_generations, "../bispectrum_train_generated_df.pth")
