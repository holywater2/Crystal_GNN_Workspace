
# from typing import Dict, List, Optional, Set, Tuple
# from jarvis.core.graphs import Graph, StructureDataset
# from sklearn.pipeline import Pipeline
# from sklearn.decomposition import PCA  # ,KernelPCA


from sklearn.preprocessing import StandardScaler

import math
import csv
import functools
import json
import os, sys
import warnings
import random
import torch
import dgl
import pandas as pd
import pickle as pk
import numpy as np

from tqdm import tqdm

from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import random_split

from jarvis.db.figshare import data
from jarvis.db.figshare import data as jdata
from jarvis.db.jsonutils import loadjson, dumpjson
from jarvis.core.atoms import Atoms
# from jarvis.core.graphs import Graph
from alignn.graphs import Graph, StructureDataset


from typing import Optional
from pathlib import Path

sys.path.append('.')
from repo_utils.sample_data import sample_data

tqdm.pandas()

#'e_hull', 'gap pbe', 'mu_b', 'elastic anisotropy', 'bulk modulus', 'shear modulus', 'e_form'

class mp:
    sample_data = sample_data
    def sample_data_as_atoms():
        return Atoms.from_dict(sample_data["atoms"])

    pdirname="/home/holywater2/2023/_Reproduce/dataset/mp_megnet"
    dirname="mp_megnet_sample001"

    def download():
        """
        Download the periodic (crystal) dataset of the Materials Project
        that is used in MEGNET by using the jarvis-Tools dataset
        
        data[0].keys()
        = dict_keys(['id', 'desc', 'formula','e_hull', 'gap pbe',
                     'mu_b', 'elastic anisotropy', 'bulk modulus',
                     'shear modulus', 'atoms', 'e_form'])
        
        Returns:
            'data'(list): the list of dictionary that handles the data of the mat
        """
        megnet = data(dataset="megnet")
        print("[I] Downloading megnet dataset is completed!")
        return megnet
    
    def save(dirname="mp_megnet_all",
             pdirname=None,
             dataset = None,
             format="json",
             sample_ratio = 1,
             id_prop = True,
             target_prop ="e_form",
             seed=123):
        """
        Save the megnet dataset
        """
        print("[I] Saving the dataset...")
        
        if dirname is None:
            dirname = "mp_megnet"
        
        """Change dierctory - default: ../dataset/mp_megnet"""
        if pdirname is None:
            pdirname = "../dataset/mp_megnet"
        path = pdirname + '/' + dirname
        print('[I] loading from',path)
        os.makedirs(pdirname,exist_ok=True)
        
        """If dirname already exist, exit the function"""
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            print("[I] Directory",path,"already exists")
            return

        cwd = os.getcwd()
        os.chdir(path)

        """If dataset is not given, download from figshare"""
        if dataset is None:
            megnet = mp.download()
        else:
            megnet = dataset
        
        """If sampling from given dataset is needed"""
        if seed is not None:
            random.seed(seed)
        
        if sample_ratio != 1:
            save_count = int(len(megnet) * sample_ratio)
            megnet = random.sample(megnet, save_count)           
        
        if format == 'json':
            for mp_item in tqdm(megnet):
                filename = mp_item["id"]+".json"
                dumpjson(data=mp_item,filename=filename)
            print("[I] mp megnet data is saved in", os.getcwd()+'/'+dirname)
        
        elif format == 'cif':
            for mp_item in tqdm(megnet):
                filename = mp_item["id"]+".cif"
                Atoms.from_dict(mp_item["atoms"]).write_cif(filename)
            print("[I] mp megnet .cif data is saved in", os.getcwd()+'/'+dirname)
        print("[I] Saving complete: the number of the saved sample is",len(megnet))
        
        """If saving id-prop.csv is true"""
        if id_prop :
            with open("id_prop.csv", 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['id','prop'])
                for mp_item in tqdm(megnet):
                    writer.writerow([mp_item['id'],mp_item[target_prop]])
            print("[I] mp megnet id_prop.csv is saved in", os.getcwd()+'/'+"id_prop.csv")
 
        os.chdir(cwd)

        return megnet


    def save_with_sample(dirname="mp_megnet_sample",
                         dataset=None,
                         format="json",
                         sample_ratio=0.1,
                         pdirname=None,
                         seed=123):
        """
        Alias of the save(sample=0.1)
        
        Args:
            dirname (str, optional): _description_. Defaults to "mg_megnet_sample".
            dataset (list, optional): _description_. Defaults to None.
            format (str, optional): _description_. Defaults to "json".
            sample (float, optional): _description_. Defaults to 0.1.
            pdirname (str, optional): _description_. Defaults to None.
        """
        mp.save(dirname=dirname,
                dataset=dataset,
                format=format,
                sample_ratio=sample_ratio,
                pdirname=pdirname,
                seed=seed)    


    def load(dirname="mp_megnet_all",
             down=False,
             downsave=True,
             format='json',
             pdirname=None):
        
        if dirname is None:
            dirname = "mp_megnet_all"
        
        if pdirname is None:
            path = "../dataset/mp_megnet/"+dirname
        else :
            path = pdirname +"/" + dirname
        

        print("[I] Loading the dataset...")
        
        if not os.path.exists(path):
            print("[I] Error: can not find", os.getcwd()+'/'+path)
            if down :
                load_megnet = mp.download()
                if downsave :
                    mp.save(path)
            else:
                print("[I] And the dataset is not downloaded since down option is False.")
                return
        else:
            load_megnet = []
            for mp_item in tqdm(os.listdir(path)):
                if os.path.splitext(mp_item)[1] == '.json':
                    load_megnet.append(loadjson(path+'/'+mp_item))
            
        #TODO : cif file load should be implemented
        print("[I] Loading complete: the number of the loaded data is",len(load_megnet))
        
        return load_megnet
    

    def load_json(dirname="mp_megnet_all",
                  down=True,
                  downsave=True,
                  pdirname=None):
        """Alias of the load(format="json")"""        
        return mp.load(dirname=dirname,
                       down=down,
                       downsave=downsave,
                       format="json",
                       pdirname=pdirname)
        
    def load_json_sample(dirname="mp_megnet_sample01",
                         down=True,
                         downsave=True,
                         pdirname=None):
        """Alias of the load(format="json")"""        
        return mp.load(dirname=dirname,
                       down=down,
                       downsave=downsave,
                       format="json",
                       pdirname=pdirname)
        
    def load_sample01():
        return mp.load(pdirname="../_Reproduce/dataset/mp_megnet",dirname="mp_megnet_sample001")
    
    def load_sample001():
        return mp.load(pdirname="../_Reproduce/dataset/mp_megnet",dirname="mp_megnet_sample001")
    
def loadjson(filename=""):
    """Provide helper function to load a json file."""
    f = open(filename, "r")
    d = json.load(f)
    f.close()
    return d


def dumpjson(data=[], filename=""):
    """Provide helper function to write a json file."""
    f = open(filename, "w")
    f.write(json.dumps(data))
    f.close()

"""Jarvis-dgl data loaders and DGLGraph utilities."""

def load_graphs(
    dataset=[],
    name: str = "dft_3d",
    neighbor_strategy: str = "k-nearest",
    cutoff: float = 8,
    max_neighbors: int = 12,
    cachedir: Optional[Path] = None,
    use_canonize: bool = False,
    supercell_dim: list = [1,1,1]
):
    """Construct crystal graphs.

    Load only atomic number node features
    and bond displacement vector edge features.

    Resulting graphs have scheme e.g.
    ```
    Graph(num_nodes=12, num_edges=156,
        ndata_schemes={'atom_features': Scheme(shape=(1,)}
        edata_schemes={'r': Scheme(shape=(3,)})
    ```
    """

    def atoms_to_graph(atoms):
        """Convert structure dict to DGLGraph."""
        structure = Atoms.from_dict(atoms)
        if supercell_dim is None or supercell_dim != [1,1,1]:
            structure = structure.make_supercell(supercell_dim)
        return Graph.atom_dgl_multigraph(
            structure,
            cutoff=cutoff,
            atom_features="atomic_number",
            max_neighbors=max_neighbors,
            compute_line_graph=False,
            use_canonize=use_canonize,
        )

    if cachedir is not None:
        cachefile = cachedir / f"{name}-{neighbor_strategy}.bin"
    else:
        cachefile = None

    if cachefile is not None and cachefile.is_file():
        graphs, labels = dgl.load_graphs(str(cachefile))
    else:
        df = pd.DataFrame(dataset)

        graphs = df["atoms"].progress_apply(atoms_to_graph).values
        if cachefile is not None:
            dgl.save_graphs(str(cachefile), graphs.tolist())

    return graphs

def load_graphs_supercell(
    dataset=[],
    name: str = "dft_3d",
    neighbor_strategy: str = "k-nearest",
    cutoff: float = 8,
    max_neighbors: int = 12,
    cachedir: Optional[Path] = None,
    use_canonize: bool = False,
    supercell_dim: list = [1,1,1]
):
    """Construct crystal graphs.

    Load only atomic number node features
    and bond displacement vector edge features.

    Resulting graphs have scheme e.g.
    ```
    Graph(num_nodes=12, num_edges=156,
        ndata_schemes={'atom_features': Scheme(shape=(1,)}
        edata_schemes={'r': Scheme(shape=(3,)})
    ```
    """

    def atoms_to_graph(atoms):
        """Convert structure dict to DGLGraph."""
        structure = Atoms.from_dict(atoms).make_supercell(supercell_dim)
        return Graph.atom_dgl_multigraph(
            structure,
            cutoff=cutoff,
            atom_features="atomic_number",
            max_neighbors=max_neighbors,
            compute_line_graph=False,
            use_canonize=use_canonize,
        )

    if cachedir is not None:
        cachefile = cachedir / f"{name}-{neighbor_strategy}.bin"
    else:
        cachefile = None

    if cachefile is not None and cachefile.is_file():
        graphs, labels = dgl.load_graphs(str(cachefile))
    else:
        df = pd.DataFrame(dataset)

        graphs = df["atoms"].progress_apply(atoms_to_graph).values
        if cachefile is not None:
            dgl.save_graphs(str(cachefile), graphs.tolist())

    return graphs

# np.mean(mean_absolute_deviation(x,axis=0))
def mean_absolute_deviation(data, axis=None):
    """Get Mean absolute deviation."""
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)

def get_id_train_val_test(
    total_size=1000,
    split_seed=123,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    n_train=None,
    n_test=None,
    n_val=None,
    keep_data_order=False,
):
    """Get train, val, test IDs."""
    if (
        train_ratio is None
        and val_ratio is not None
        and test_ratio is not None
    ):
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print("Using rest of the dataset except the test and val sets.")
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    # indices = list(range(total_size))
    if n_train is None:
        n_train = int(train_ratio * total_size)
    if n_test is None:
        n_test = int(test_ratio * total_size)
    if n_val is None:
        n_val = int(val_ratio * total_size)
    ids = list(np.arange(total_size))
    if not keep_data_order:
        random.seed(split_seed)
        random.shuffle(ids)
    # np.random.shuffle(ids)
    if n_train + n_val + n_test > total_size:
        raise ValueError(
            "Check total number of samples.",
            n_train + n_val + n_test,
            ">",
            total_size,
        )

    # shuffle consistently with https://github.com/txie-93/cgcnn/data.py
    # i.e. shuffle the index in place with standard library random.shuffle
    # first obtain only valid indices

    # test_size = round(N * 0.2)

    # full train/val test split
    # ids = ids[::-1]
    id_train = ids[:n_train]
    id_val = ids[-(n_val + n_test) : -n_test]  # noqa:E203
    id_test = ids[-n_test:]
    return id_train, id_val, id_test


def get_torch_dataset(
    dataset=[],
    id_tag="jid",
    target="",
    target_atomwise="",
    target_grad="",
    target_stress="",
    neighbor_strategy="",
    atom_features="",
    use_canonize="",
    name="",
    line_graph="",
    cutoff=8.0,
    max_neighbors=12,
    classification=False,
    output_dir=".",
    tmp_name="dataset",
    supercell_dim = [1,1,1]
):
    """Get Torch Dataset."""
    df = pd.DataFrame(dataset)
    # df['natoms']=df['atoms'].apply(lambda x: len(x['elements']))
    # print(" data df", df)
    vals = np.array([ii[target] for ii in dataset])  # df[target].values
    print("data range", np.max(vals), np.min(vals))
    f = open(os.path.join(output_dir, tmp_name + "_data_range"), "w")
    line = "Max=" + str(np.max(vals)) + "\n"
    f.write(line)
    line = "Min=" + str(np.min(vals)) + "\n"
    f.write(line)
    f.close()

    graphs = load_graphs(
        df,
        name=name,
        neighbor_strategy=neighbor_strategy,
        use_canonize=use_canonize,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        supercell_dim=supercell_dim
    )
    data = StructureDataset(
        df,
        graphs,
        target=target,
        target_atomwise=target_atomwise,
        target_grad=target_grad,
        target_stress=target_stress,
        atom_features=atom_features,
        line_graph=line_graph,
        id_tag=id_tag,
        classification=classification,
    )
    return data


def get_train_val_loaders(
    dataset: str = "dft_3d",
    dataset_array=[],
    target: str = "formation_energy_peratom",
    target_atomwise: str = "",
    target_grad: str = "",
    target_stress: str = "",
    atom_features: str = "cgcnn",
    neighbor_strategy: str = "k-nearest",
    n_train=None,
    n_val=None,
    n_test=None,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    batch_size: int = 5,
    standardize: bool = False,
    line_graph: bool = True,
    split_seed: int = 123,
    workers: int = 0,
    pin_memory: bool = True,
    save_dataloader: bool = False,
    filename: str = "sample",
    id_tag: str = "jid",
    use_canonize: bool = False,
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    classification_threshold: Optional[float] = None,
    target_multiplication_factor: Optional[float] = None,
    standard_scalar_and_pca=False,
    keep_data_order=False,
    output_features=1,
    output_dir=None,
    supercell_dim: list=[1,1,1]
):
    """Help function to set up JARVIS train and val dataloaders."""
    train_sample = filename + "_train.data"
    val_sample = filename + "_val.data"
    test_sample = filename + "_test.data"
    # print ('output_dir data',output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if (
        os.path.exists(train_sample)
        and os.path.exists(val_sample)
        and os.path.exists(test_sample)
        and save_dataloader
    ):
        print("Loading from saved file...")
        print("Make sure all the DataLoader params are same.")
        print("This module is made for debugging only.")
        train_loader = torch.load(train_sample)
        val_loader = torch.load(val_sample)
        test_loader = torch.load(test_sample)
        if train_loader.pin_memory != pin_memory:
            train_loader.pin_memory = pin_memory
        if test_loader.pin_memory != pin_memory:
            test_loader.pin_memory = pin_memory
        if val_loader.pin_memory != pin_memory:
            val_loader.pin_memory = pin_memory
        if train_loader.num_workers != workers:
            train_loader.num_workers = workers
        if test_loader.num_workers != workers:
            test_loader.num_workers = workers
        if val_loader.num_workers != workers:
            val_loader.num_workers = workers
        # print("train", len(train_loader.dataset))
        # print("val", len(val_loader.dataset))
        # print("test", len(test_loader.dataset))
    else:

        if not dataset_array:
            d = jdata(dataset)
        else:
            d = dataset_array

            # for ii, i in enumerate(pc_y):
            #    d[ii][target] = pc_y[ii].tolist()

        dat = []
        if classification_threshold is not None:
            print(
                "Using ",
                classification_threshold,
                " for classifying ",
                target,
                " data.",
            )
            print("Converting target data into 1 and 0.")
        all_targets = []

        # TODO:make an all key in qm9_dgl
        if dataset == "qm9_dgl" and target == "all":
            print("Making all qm9_dgl")
            tmp = []
            for ii in d:
                ii["all"] = [
                    ii["mu"],
                    ii["alpha"],
                    ii["homo"],
                    ii["lumo"],
                    ii["gap"],
                    ii["r2"],
                    ii["zpve"],
                    ii["U0"],
                    ii["U"],
                    ii["H"],
                    ii["G"],
                    ii["Cv"],
                ]
                tmp.append(ii)
            print("Made all qm9_dgl")
            d = tmp
        for i in d:
            if isinstance(i[target], list):  # multioutput target
                all_targets.append(torch.tensor(i[target]))
                dat.append(i)

            elif (
                i[target] is not None
                and i[target] != "na"
                and not math.isnan(i[target])
            ):
                if target_multiplication_factor is not None:
                    i[target] = i[target] * target_multiplication_factor
                if classification_threshold is not None:
                    if i[target] <= classification_threshold:
                        i[target] = 0
                    elif i[target] > classification_threshold:
                        i[target] = 1
                    else:
                        raise ValueError(
                            "Check classification data type.",
                            i[target],
                            type(i[target]),
                        )
                dat.append(i)
                all_targets.append(i[target])

        # id_test = ids[-test_size:]
        # if standardize:
        #    data.setup_standardizer(id_train)
        id_train, id_val, id_test = get_id_train_val_test(
            total_size=len(dat),
            split_seed=split_seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            n_train=n_train,
            n_test=n_test,
            n_val=n_val,
            keep_data_order=keep_data_order,
        )
        ids_train_val_test = {}
        ids_train_val_test["id_train"] = [dat[i][id_tag] for i in id_train]
        ids_train_val_test["id_val"] = [dat[i][id_tag] for i in id_val]
        ids_train_val_test["id_test"] = [dat[i][id_tag] for i in id_test]
        dumpjson(
            data=ids_train_val_test,
            filename=os.path.join(output_dir, "ids_train_val_test.json"),
        )
        dataset_train = [dat[x] for x in id_train]
        dataset_val = [dat[x] for x in id_val]
        dataset_test = [dat[x] for x in id_test]

        if standard_scalar_and_pca:
            y_data = [i[target] for i in dataset_train]
            # pipe = Pipeline([('scale', StandardScaler())])
            if not isinstance(y_data[0], list):
                print("Running StandardScalar")
                y_data = np.array(y_data).reshape(-1, 1)
            sc = StandardScaler()

            sc.fit(y_data)
            print("Mean", sc.mean_)
            print("Variance", sc.var_)
            try:
                print("New max", max(y_data))
                print("New min", min(y_data))
            except Exception as exp:
                print(exp)
                pass
            # pc = PCA(n_components=output_features)
            # pipe = Pipeline(
            #    [
            #        ("scale", StandardScaler()),
            #        ("reduce_dims", PCA(n_components=output_features)),
            #    ]
            # )
            pk.dump(sc, open(os.path.join(output_dir, "sc.pkl"), "wb"))
            # pc = PCA(n_components=10)
            # pc.fit(y_data)
            # pk.dump(pc, open("pca.pkl", "wb"))

        if classification_threshold is None:
            try:
                from sklearn.metrics import mean_absolute_error

                print("MAX val:", max(all_targets))
                print("MIN val:", min(all_targets))
                print("MAD:", mean_absolute_deviation(all_targets))
                try:
                    f = open(os.path.join(output_dir, "mad"), "w")
                    line = "MAX val:" + str(max(all_targets)) + "\n"
                    line += "MIN val:" + str(min(all_targets)) + "\n"
                    line += (
                        "MAD val:"
                        + str(mean_absolute_deviation(all_targets))
                        + "\n"
                    )
                    f.write(line)
                    f.close()
                except Exception as exp:
                    print("Cannot write mad", exp)
                    pass
                # Random model precited value
                x_bar = np.mean(np.array([i[target] for i in dataset_train]))
                baseline_mae = mean_absolute_error(
                    np.array([i[target] for i in dataset_test]),
                    np.array([x_bar for i in dataset_test]),
                )
                print("Baseline MAE:", baseline_mae)
            except Exception as exp:
                print("Data error", exp)
                pass

        train_data = get_torch_dataset(
            dataset=dataset_train,
            id_tag=id_tag,
            atom_features=atom_features,
            target=target,
            target_atomwise=target_atomwise,
            target_grad=target_grad,
            target_stress=target_stress,
            neighbor_strategy=neighbor_strategy,
            use_canonize=use_canonize,
            name=dataset,
            line_graph=line_graph,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            classification=classification_threshold is not None,
            output_dir=output_dir,
            tmp_name="train_data",
            supercell_dim=supercell_dim
        )
        val_data = get_torch_dataset(
            dataset=dataset_val,
            id_tag=id_tag,
            atom_features=atom_features,
            target=target,
            target_atomwise=target_atomwise,
            target_grad=target_grad,
            target_stress=target_stress,
            neighbor_strategy=neighbor_strategy,
            use_canonize=use_canonize,
            name=dataset,
            line_graph=line_graph,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            classification=classification_threshold is not None,
            output_dir=output_dir,
            tmp_name="val_data",
            supercell_dim=supercell_dim
        )
        test_data = get_torch_dataset(
            dataset=dataset_test,
            id_tag=id_tag,
            atom_features=atom_features,
            target=target,
            target_atomwise=target_atomwise,
            target_grad=target_grad,
            target_stress=target_stress,
            neighbor_strategy=neighbor_strategy,
            use_canonize=use_canonize,
            name=dataset,
            line_graph=line_graph,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            classification=classification_threshold is not None,
            output_dir=output_dir,
            tmp_name="test_data",
            supercell_dim=supercell_dim
        )

        collate_fn = train_data.collate
        # print("line_graph,line_dih_graph", line_graph, line_dih_graph)
        if line_graph:
            collate_fn = train_data.collate_line_graph

        # use a regular pytorch dataloader
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=workers,
            pin_memory=pin_memory,
        )

        val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=workers,
            pin_memory=pin_memory,
        )

        test_loader = DataLoader(
            test_data,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
            num_workers=workers,
            pin_memory=pin_memory,
        )
        if save_dataloader:
            torch.save(train_loader, train_sample)
            torch.save(val_loader, val_sample)
            torch.save(test_loader, test_sample)
    print("n_train:", len(train_loader.dataset))
    print("n_val:", len(val_loader.dataset))
    print("n_test:", len(test_loader.dataset))
    return (
        train_loader,
        val_loader,
        test_loader,
        train_loader.dataset.prepare_batch,
    )

def sample_train_test_val(dirname="mp_megnet_sample001",
                          target='e_form',
                          batchsize=64,
                          filename="./sample/loader",
                          save_dataloader="./sample/save_loader",
                          output_dir="./sample/output_dir",
                          max_neighbors=12,
                          num_workers=8,
                          split_seed=123,
                          radial_cutoff=5.0,
                          supercell_dim=[1,1,1]):
    dataset = mp.load_json_sample(dirname,pdirname=mp.pdirname)
    return get_train_val_loaders(dataset="megnet",
                            dataset_array=dataset,
                            target=target,
                            batch_size=batchsize,
                            filename=filename,
                            save_dataloader=save_dataloader,
                            output_dir=output_dir,
                            line_graph=False,
                            max_neighbors=max_neighbors,
                            workers=num_workers,
                            split_seed=split_seed,
                            cutoff=radial_cutoff,
                            # atom_features=,
                            id_tag="id",
                            supercell_dim=supercell_dim)

if __name__ == "__main__":
    megnet = mp.load(down=True,downsave=True)
    mp.save(dataset=megnet)
    mp.save_with_sample(dataset=megnet,
                    dirname="mp_megnet_sample001",
                    sample_ratio=0.01)
    mp.save_with_sample(dataset=megnet,
                    dirname="mp_megnet_sample01",
                    sample_ratio=0.1)
    