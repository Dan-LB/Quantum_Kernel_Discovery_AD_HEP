import h5py

from darqk.oneclassqsvm.data_processing import get_data

filename = "latentrep_QCD_sig.h5"
filename = "latentrep_QCD_sig_testclustering.h5"

lista = ["latentrep_QCD_sig.h5",
         "latentrep_QCD_sig_testclustering.h5",
         "latentrep_AtoHZ_to_ZZZ_35.h5",
         "latentrep_RSGraviton_WW_BR_15.h5",
         "latentrep_RSGraviton_WW_NA_35.h5"
         ]

for filename in lista:
    with h5py.File(filename, "r") as f:
        # Print all root level object names (aka keys) 
        # these can be group or dataset names 
        print("Keys: %s" % f.keys())
        # get first object name/key; may or may NOT be a group
        a_group_key = list(f.keys())[0]

        # get the object type for a_group_key: usually group or dataset
        print(type(f[a_group_key])) 

        print(a_group_key)

        # If a_group_key is a group name, 
        # this gets the object names in the group and returns as a list
        data = list(f[a_group_key])


        #print(data)

        # preferred methods to get dataset values:
        ds_obj = f[a_group_key]      # returns as a h5py dataset object

    # print(ds_obj)

        ds_arr = f[a_group_key][()]  # returns as a numpy array

        print(ds_arr)
        
x = 1/0

args = {'sig_path': 'latentrep_AtoHZ_to_ZZZ_35.h5', 
            'bkg_path': 'latentrep_QCD_sig.h5',
            'test_bkg_path': 'latentrep_QCD_sig_testclustering.h5',
            'unsup': True,
            'nqubits' : 4,
            'feature_map': 'u_dense_encoding',
            'run_type' : 'ideal',
            'output_folder' : 'quantum_test',
            'nu_param' :'0.01',
            'ntrain':600,
            'quantum': True,
            'ntest':10 
            }

train_loader, test_loader = get_data(args)
X_train, y_train = train_loader[0], train_loader[1]
print(f'train_loader 0: {X_train}')
print(f'train_loader 1: {y_train}')