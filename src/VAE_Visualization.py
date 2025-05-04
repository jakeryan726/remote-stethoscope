import torch
from models import VAE
import umap
import matplotlib.pyplot as plt
import plotly.express as px


def visualize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dl = torch.load("../bispectrum_train_dl.pt")
    model = torch.load("../saved_models/VAE_model.pt")
    original_mapping = torch.load("../saved_models/Encoder_mapping.pt")
    mapping = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    model.eval()
    with torch.no_grad():
        vectors = []
        labels = []
        for batch in dl:
            x, y = batch
            x = x.to(device)
            mu, logvar = model.encode(x)
            lv = model.reparameterize(mu, logvar)
            vectors.append(lv)
            labels += [mapping[idx] for idx in torch.argmax(y, dim=1)]
    vectors = torch.cat(vectors, dim=0)

    reducer = umap.UMAP(n_components=3)
    embedding = reducer.fit_transform(vectors.cpu())

    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=labels)
    plt.colorbar(scatter, label="Labels")
    plt.title("VAE Latent Space UMAP Projection")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")

    handles, _ = scatter.legend_elements()
    plt.legend(handles, original_mapping, title="Categories")
    plt.savefig("../plots/2d_UMAP_plot.png", dpi=300, bbox_inches="tight")

    fig_3d = px.scatter_3d(
        embedding, x=0, y=1, z=2, color=labels, labels={"color": "Diagnosis"}
    )
    fig_3d.update_traces(marker_size=5)
    fig_3d.write_html("../plots/3d_UMAP_plot.html")


if __name__ == "__main__":
    visualize()
