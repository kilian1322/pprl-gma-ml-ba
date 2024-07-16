from main import run

GLOBAL_CONFIG = {
    "Data": "./data/Daten_PPRL/fakename_1k.tsv",
    "Overlap": 0.9,
    "DropFrom": "Both",
    "DevMode": False,  # Development Mode, saves some intermediate results to the /dev directory
    "BenchMode": True,  # Benchmark Mode
    "Verbose": False,  # Print Status Messages?
    "MatchingMetric": "cosine",
    "Matching": "Symmetric",
    "Workers": -1,
    "Data_error": 0         #"./data/Daten_PPRL/fakename_1k.tsv"
}

ENC_CONFIG = {
    "AliceAlgo": "BloomFilter",
    "AliceSecret": "SuperSecretSalt1337",
    "AliceN": 2,
    "AliceMetric": "dice",
    "EveAlgo": "BloomFilter",                                 
    "EveSecret": "SuperSecretSalt1337",
    "EveN": 2,
    "EveMetric": "dice",
    # For BF encoding
    "AliceBFLength": 1024,
    "AliceBits": 10,
    "AliceDiffuse": False,                    
    "AliceT": 10,
    "AliceEldLength": 1024,
    "EveBFLength": 1024,
    "EveBits": 10,
    "EveDiffuse": False,                      
    "EveT": 10,
    "EveEldLength": 1024,
    # For TMH encoding
    "AliceNHash": 1024,
    "AliceNHashBits": 64,
    "AliceNSubKeys": 8,
    "Alice1BitHash": True,
    "EveNHash": 2000,
    "EveNHashBits": 32,
    "EveNSubKeys": 8,
    "Eve1BitHash": True,
    # For 2SH encoding
    "AliceNHashFunc": 10,
    "AliceNHashCol": 1000,
    "AliceRandMode": "PNG",
    "EveNHashFunc": 10,
    "EveNHashCol": 1000,
    "EveRandMode": "PNG"
}

EMB_CONFIG = {
    "Algo": "Node2Vec",
    "AliceQuantile": 0.9,
    "AliceDiscretize": False,
    "AliceDim": 128,
    "AliceContext": 10,
    "AliceNegative": 1,
    "AliceNormalize": True,
    "EveQuantile": 0.9,
    "EveDiscretize": False,
    "EveDim": 128,
    "EveContext": 10,
    "EveNegative": 1,
    "EveNormalize": True,
    # For Node2Vec
    "AliceWalkLen": 100,
    "AliceNWalks": 20,
    "AliceP": 250,  # 0.5
    "AliceQ": 300,  # 2z
    "AliceEpochs": 5,
    "AliceSeed": 42,
    "EveWalkLen": 100,
    "EveNWalks": 20,
    "EveP": 250,  # 0.5
    "EveQ": 300,  # 2
    "EveEpochs": 5,
    "EveSeed": 42
}

ALIGN_CONFIG = {
    "RegWS": GLOBAL_CONFIG["Overlap"] / 2, 
    "RegInit": 1,  # For BF 0.25
    "Batchsize": 1,  # 1 = 100%
    "LR": 200.0,
    "NIterWS": 20,
    "NIterInit": 5,  
    "NEpochWS": 100,
    "LRDecay": 0.999,
    "Sqrt": True,
    "EarlyStopping": 10,
    "Selection": "None",
    "MaxLoad": None,
    "Wasserstein": True,
    "Dummy": 0
}
# Global params
datasets = ["fakename_1k.tsv", "fakename_2k.tsv", "fakename_5k.tsv", "fakename_10k.tsv", "titanic_full.tsv"] 
overlap = [i/100 for i in range(5, 105, 5)]
discretize = [False]
supported_encs = ["BloomFilter"]    #TabMinHash", "TwoStepHash"]  
supported_emb = "Node2Vec"
supported_allignment = "Wasserstein"
#supported_matchings = ["MinWeight", "Stable", "Symmetric", "NearestNeighbor"]
subsets = ["Both"] #["Both", "Alice"]
#dummy =[10, 100, 1000, 2000, 5000]
""" error_dict = {
              'fakename_2k.tsv': ["fakename_2k_error_20.tsv"], 
              'fakename_5k.tsv': ["fakename_5k_error_2.tsv", "fakename_5k_error_5.tsv", "fakename_5k_error_10.tsv", "fakename_5k_error_15.tsv", "fakename_5k_error_20.tsv"],
              'fakename_10k.tsv': ["fakename_10k_error_2.tsv", "fakename_10k_error_5.tsv", "fakename_10k_error_10.tsv", "fakename_10k_error_15.tsv", "fakename_10k_error_20.tsv"]
              } """



for subset in subsets:
    GLOBAL_CONFIG["DropFrom"] = subset

    #for d in datasets:
    for d in datasets:
        GLOBAL_CONFIG["Data"] = "./data/Daten_PPRL/" + d
        
        for o in overlap:
            GLOBAL_CONFIG["Overlap"] = o
            ALIGN_CONFIG["RegWS"] = max(0.1, o/2)
            
                
            print(ENC_CONFIG["AliceAlgo"])
            print(subset) 
            print(d)
            print(o)
            print(ALIGN_CONFIG["Selection"])
            #print(ALIGN_CONFIG["Dummy"])
            print(GLOBAL_CONFIG["Matching"])
            print()
        
            run(GLOBAL_CONFIG.copy(), ENC_CONFIG.copy(), EMB_CONFIG.copy(), ALIGN_CONFIG.copy())
            print("\n")
