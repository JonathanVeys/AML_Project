import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchmetrics import F1Score

def train_and_cache():
    ############################
    # load data
    ROOT = Path(__file__).resolve().parents[2]
    train_data = np.load(ROOT / "data/species_train.npz", allow_pickle=True)
    print(train_data)
    species_data = pd.DataFrame(
        {
            'ID':train_data['train_ids'],
            'Longitude':train_data['train_locs'][:,0],
            'Latitude':train_data['train_locs'][:,1]
        }
    )
    
    print(len(species_data))

    taxon_names_lookup = pd.DataFrame(
        {
            'ID':train_data['taxon_ids'],
            'Names':train_data['taxon_names']
        }
    )
    
    print(len(taxon_names_lookup))

    number_of_classes = len(train_data["taxon_ids"])
    assert number_of_classes == 500
    print("OK")

    # each row represents one location and each column represents a species ID
    # there is exactly same row
    tmp = species_data.copy()
    tmp["row_id"] = tmp.index
    grid = pd.crosstab(
        index=[tmp["Longitude"], tmp["Latitude"], tmp["row_id"]],
        columns=tmp["ID"].astype(int)
    )

    # check if each location have only one corresponding species
    row_sums = grid.sum(axis=1).to_numpy()
    assert np.all(row_sums == 1)

    size_of_grid = len(grid)
    grid = grid.droplevel("row_id")

    # ensure drop will not change the table
    assert len(grid) == size_of_grid
    
    longitude_latitude = grid.index.to_frame()[["Longitude","Latitude"]].to_numpy()
    y = torch.from_numpy(grid.to_numpy(np.float32))
    
    ############################
    # preprocess
    # encode the gps coordinate into the sin and cos form as others code
    # the pytorch build in fucntion can not work here
    longitude = torch.tensor(longitude_latitude[:,0]) * torch.pi/180
    latitude = torch.tensor(longitude_latitude[:,1]) * torch.pi/180

    X_processed = torch.stack(
        [torch.sin(longitude), torch.cos(longitude), torch.sin(latitude), torch.cos(latitude)],
    dim=1).float()
    
    # split data into train, eval and test
    total_number = X_processed.shape[0]
    assert total_number == 272037
    labels = y.argmax(dim=1).long()

    train_and_eval_ids, test_ids = train_test_split(
        np.arange(total_number),
        test_size=0.1,
        random_state=77,
        stratify=labels.cpu().numpy()
    )
    
    train_ids, eval_ids = train_test_split(
        train_and_eval_ids,
        test_size=0.1,
        random_state=77,
        stratify=labels.cpu().numpy()[train_and_eval_ids]
    )
    
    X_train = X_processed[train_ids]
    y_train = labels[train_ids]
    X_test = X_processed[test_ids]
    y_test = labels[test_ids]
    X_eval = X_processed[eval_ids]
    y_eval = labels[eval_ids]
    
    ############################
    # my neural network
    number_of_input_units = 4
    # the number of units from 1000 to 300 doesnt matter much
    number_of_first_layer_units = 400
    number_of_output_layer_units = number_of_classes
    
    # 3 hidden layers is good enough
    # there isn't much improvement by using 4 hidden layers
    net = nn.Sequential(
        nn.Linear(number_of_input_units, number_of_first_layer_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(number_of_first_layer_units, number_of_first_layer_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(number_of_first_layer_units, number_of_first_layer_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(number_of_first_layer_units, number_of_output_layer_units),
    )

    learning_rate = 4e-3
    optimiser = torch.optim.Adam(net.parameters(), learning_rate, weight_decay=5e-4)

    loss_function = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=8192, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=1024, shuffle=False)
    eval_loader = DataLoader(TensorDataset(X_eval, y_eval), batch_size=2048, shuffle=False)
    
    ############################
    # trainning
    # normally training will stop much ealier than 100
    number_of_epoch = 100
    
    # how much continus times that is allowed to have no improvements
    patience = 3
    # how small the improvement is allowed
    improve_limitation = 1e-4
    best_eval = float('inf') # a positive infinity
    epochs_no_improve = 0
    best_state = None
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', factor=0.5, patience=1,
        threshold=1e-4, cooldown=0, min_lr=0.0
    )

    for i in range(number_of_epoch):
        # more epoch seems work better
        net.train()
        for X_i, y_i in train_loader:
            optimiser.zero_grad()
            result_y = net(X_i)
            loss = loss_function(result_y, y_i)
            print(loss)
            loss.backward()
            optimiser.step()
        
        net.eval()
        eval_loss_sum, eval_correct, eval_total = 0.0, 0, 0
        with torch.inference_mode():
            for X_i, y_i in eval_loader:
                result_y = net(X_i)
                eval_loss = loss_function(result_y, y_i)
                eval_loss_sum += eval_loss.item() * y_i.size(0)
                pred = result_y.argmax(1)
                # expect the result is the same as the label
                # top1
                eval_correct += (pred == y_i).sum().item()
                eval_total += y_i.size(0)
        eval_ce = eval_loss_sum / eval_total
        eval_top1 = eval_correct / eval_total
        print(f"[Epoch {i}:{number_of_epoch}] | Evaluation Cross Entropy: {eval_ce:.4f} | Top-1: {eval_correct}/{eval_total},{eval_top1:.4f}")

        scheduler.step(eval_ce)

        if best_eval - eval_ce > improve_limitation:
            best_eval = eval_ce
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            print(f"*** New best eval CE: {best_eval:.4f} (epoch {i}:{number_of_epoch}) ***")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{patience} epochs")
            if epochs_no_improve >= patience:
                print("*** Early Stopping Triggered ***")
                break
    
    # use the best parameters
    if best_state is not None:
        net.load_state_dict(best_state)
    else:
        print("***Warning*** WTF????")

    ############################
    # testing
    softmax = nn.Softmax(dim=1)

    net.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    test_ce_sum = 0.0
    
    with torch.inference_mode():
        for X_i, y_i in test_loader:
            result_y = net(X_i)

            # loss
            loss = loss_function(result_y, y_i)
            test_ce_sum += loss.item() * y_i.size(0)

            # softmax
            probs = softmax(result_y)
            top1 = probs.argmax(dim=1)

            # accuracies
            correct_top1 += (top1 == y_i).sum().item()
            top5 = probs.topk(5, dim=1).indices
            correct_top5 += (top5 == y_i.unsqueeze(1)).any(dim=1).sum().item()

            total += y_i.size(0)

    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total
    avg_test_ce = test_ce_sum / total

    print(f"Test Top-1 Accuracy: {correct_top1}/{total},{top1_acc:.4f}")
    print(f"Test Top-5 Accuracy: {correct_top5}/{total},{top5_acc:.4f}")
    print(f"Avg Test Cross-Entropy: {avg_test_ce:.4f}")

    # cache result and data set
    outdir = ROOT / "artifacts"
    outdir.mkdir(parents=True, exist_ok=True)

    # save model
    net.eval()
    example = torch.zeros(1, 4)
    ts = torch.jit.trace(net, example)
    (ts_path := outdir / "model.pt")
    ts.save(str(ts_path))

    # save corresponding data set
    np.savez_compressed(
        outdir / "test_data_cache.npz",
        X_test=X_test.cpu().numpy().astype(np.float32),
        y_test=y_test.cpu().numpy().astype(np.int64),
    )

    print(f"Saved model -> {ts_path}")
    print(f"Saved test data -> {outdir/'test_data_cache.npz'}")


def figure_and_metrix(net, test_loader, number_of_classes, topN: int = 100):
    ############################
    # evaluate performance and generate a bar diagram
    # focus on top N species
    f1_macro = F1Score(task="multiclass", num_classes=number_of_classes, average="macro")
    f1_weighted = F1Score(task="multiclass", num_classes=number_of_classes, average="weighted")

    class_count = torch.zeros(number_of_classes, dtype=torch.long) # true count per class for recall
    class_correct1 = torch.zeros(number_of_classes, dtype=torch.long) # top1 true positives
    class_correct5 = torch.zeros(number_of_classes, dtype=torch.long) # top5 true positives
    
    softmax = nn.Softmax(dim=1)

    net.eval()
    with torch.inference_mode():
        for X_i, y_i in test_loader:
            result_y = net(X_i)
    
            # f1
            f1_macro.update(result_y, y_i)
            f1_weighted.update(result_y, y_i)

            # softmax
            probs = softmax(result_y)
            top1 = probs.argmax(dim=1)
            top5 = probs.topk(5, dim=1).indices

            # counts for recall
            class_count += torch.bincount(y_i, minlength=number_of_classes)

            # true positives for top1 and top5
            correct_top1 = top1.eq(y_i)
            class_correct1 += torch.bincount(y_i[correct_top1], minlength=number_of_classes)

            correct_top5 = (top5 == y_i.unsqueeze(1)).any(dim=1)
            class_correct5 += torch.bincount(y_i[correct_top5], minlength=number_of_classes)

    # f1
    test_f1_macro = float(f1_macro.compute())
    test_f1_weighted = float(f1_weighted.compute())
    print(f"Test F1 (macro): {test_f1_macro:.4f}")
    print(f"Test F1 (weighted): {test_f1_weighted:.4f}")

    # recall for each classes
    denom_true = class_count.clamp_min(1).float()
    recall_top1 = (class_correct1.float() / denom_true).cpu().numpy()
    recall_top5 = (class_correct5.float() / denom_true).cpu().numpy()

    # accuracy
    total_samples = int(class_count.sum().item())
    overall_top1 = float(class_correct1.sum().item() / total_samples)
    overall_top5 = float(class_correct5.sum().item() / total_samples)
    print(f"Test Top1: {overall_top1:.4f}")
    print(f"Test Top5: {overall_top5:.4f}")

    # means
    mean_recall_top1 = float(np.mean(recall_top1))
    mean_recall_top5 = float(np.mean(recall_top5))

    ############################
    # bar diagram: per-class recall + per-class accuracy
    freq_idx = torch.argsort(class_count, descending=True)[:topN].cpu().numpy()
    x = np.arange(topN)
    group_width = 0.6
    bar_width = group_width / 2.0

    _, ax = plt.subplots(figsize=(max(0.1 * topN, 22), 6))
    ax.bar(x - bar_width/2, recall_top1[freq_idx], width=bar_width, label="Per-class Top-1 Recall")
    ax.bar(x + bar_width/2, recall_top5[freq_idx], width=bar_width, label="Per-class Top-5 Recall")
    
    ax.set_xticks(x)
    ax.set_xticklabels(freq_idx, rotation=90, ha='center')
    ax.set_ylabel("Recall")
    ax.set_title(f"Per-class Recall (Top-{topN} frequent classes)")

    # mean recall and overall accuracy lines
    ax.axhline(mean_recall_top1, linestyle="--", linewidth=2, color='orange',
               label=f"Mean Top-1 Recall = {mean_recall_top1:.4f}")
    ax.axhline(mean_recall_top5, linestyle=":", linewidth=2, color='green',
               label=f"Mean Top-5 Recall = {mean_recall_top5:.4f}")
    ax.axhline(overall_top1, linestyle="-.", linewidth=2, color='red',
               label=f"Overall Top-1 Accuracy = {overall_top1:.4f}")
    ax.axhline(overall_top5, linestyle="-", linewidth=2, color='blue',
               label=f"Overall Top-5 Accuracy = {overall_top5:.4f}")

    ax.legend(ncol=3, loc='upper right')

    # make the y axis visible
    plt.subplots_adjust(left=0.035, right=0.95, bottom=0.2, top=0.9)
    ax.tick_params(axis='y', length=6, width=1.2, direction='inout')
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    plt.show()


def test_on_trained_model():
    ############################
    # reload data and model
    ROOT = Path(__file__).resolve().parents[2]
    artifacts = ROOT / "artifacts"

    model_path = artifacts / "model.pt"
    data_path  = artifacts / "test_data_cache.npz"

    net = torch.jit.load(str(model_path), map_location="cpu")
    # i hope this hard code 500 will not cause any errors
    number_of_classes = 500

    d = np.load(str(data_path))
    X_test = torch.from_numpy(d["X_test"])
    y_test = torch.from_numpy(d["y_test"])
    test_loader = DataLoader(TensorDataset(X_test, y_test.long()), batch_size=2048, shuffle=False)
    
    figure_and_metrix(net, test_loader, number_of_classes, topN=100)


def test_on_real_test_data():
    ############################
    # reload model
    ROOT = Path(__file__).resolve().parents[2]
    artifacts = ROOT / "artifacts"

    model_path = artifacts / "model.pt"
    net = torch.jit.load(str(model_path), map_location="cpu")
    net.eval()

    ############################
    # load train data to recover class ID ordering
    train_data = np.load(ROOT / "data/species_train.npz", allow_pickle=True)
    class_ids = np.sort(train_data["taxon_ids"].astype(int))
    number_of_classes = len(class_ids)
    assert number_of_classes == 500

    # record a map from id to index of model output
    id_to_index = {}
    for index in range(len(class_ids)):
        id = int(class_ids[index])
        id_to_index[id] = index

    ############################
    # load real test data
    data_test = np.load(ROOT / "data/species_test.npz", allow_pickle=True)
    test_locs = data_test["test_locs"]
    test_taxon_ids = data_test["taxon_ids"].astype(int)
    test_pos_inds_raw = data_test["test_pos_inds"]

    assert set(class_ids.tolist()) == set(test_taxon_ids.tolist()), \
        "fundamental failure!!! they give us a trash test set!!!"
    
    ############################
    # preprocess
    # encode the gps coordinate into the sin and cos form as others code
    # the pytorch build in fucntion can not work here
    longitude = torch.tensor(test_locs[:,0]) * torch.pi/180
    latitude = torch.tensor(test_locs[:,1]) * torch.pi/180
    
    ############################
    # modfiy the original data structure to (location,species)
    X_test_processed = torch.stack(
        [torch.sin(longitude), torch.cos(longitude), torch.sin(latitude), torch.cos(latitude)],
    dim=1).float()
    
    X_list = []
    y_list = []

    for id, pos_index in zip(test_taxon_ids, test_pos_inds_raw):
        id = int(id)
        class_index = id_to_index[id]

        for loc_idx in pos_index:
            X_list.append(X_test_processed[loc_idx])
            y_list.append(class_index)

    X_test = torch.stack(X_list, dim=0)
    y_test = torch.tensor(y_list, dtype=torch.long)

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=2048, shuffle=False)
    
    figure_and_metrix(net, test_loader, number_of_classes, topN=100)


def check_test_data():
    ROOT = Path(__file__).resolve().parents[2]
    data_test = np.load(ROOT / "data/species_test.npz", allow_pickle=True)

    test_locs = data_test["test_locs"]
    test_taxon_ids = data_test["taxon_ids"]
    test_pos_inds_raw = data_test["test_pos_inds"]

    locations = test_locs.shape[0]
    number_of_species = len(test_taxon_ids)
    print("locations =", locations)
    print("number_of_species =", number_of_species)
    print(test_pos_inds_raw.shape)

    # count how many species are present at each test location
    species_count_per_loc = np.zeros(locations, dtype=np.int32)

    for pos_index in test_pos_inds_raw:
        # pos_index is a array stores where the species occur
        species_count_per_loc[pos_index] += 1

    num_zero = np.sum(species_count_per_loc == 0)
    num_one = np.sum(species_count_per_loc == 1)
    num_two = np.sum(species_count_per_loc == 2)
    num_three = np.sum(species_count_per_loc == 3)
    num_four = np.sum(species_count_per_loc == 4)
    num_five = np.sum(species_count_per_loc > 4)

    print("locations with 0 species:", num_zero)
    print("locations with exactly 1 species:", num_one)
    print("locations with exactly 2 species:", num_two)
    print("locations with exactly 3 species:", num_three)
    print("locations with exactly 4 species:", num_four)
    print("locations with > 4:", num_five )


if __name__ == '__main__':
    # if the model have been trained and you dont wann train again
    # just comment out the function below
    train_and_cache()
    test_on_trained_model()
    test_on_real_test_data()
    # check_test_data()