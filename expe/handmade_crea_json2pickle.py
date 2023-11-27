import contextlib
import fractions
import json
import os
import re
import shutil
import urllib
import torch
import torchvision.transforms.functional as Fvision
from sensorimotor_lenia.system import TmpLenia


rescale_to_40x40 = False
init_hflip = True
save_animals = True
prefilter = False
if prefilter:
    final_step = 500
    low_mass_threshold = 0.0
    high_mass_threshold = 6400.0

def rle2arr(st):
    '''
    Transforms an RLE string to a numpy array.

    Code from Bert Chan.

    :param st Description of the array in RLE format.
    :return Numpy array.
    '''

    rle_groups = re.findall("(\d*)([p-y]?[.boA-X$])", st.rstrip('!'))  # [(2 yO)(1 $)(1 yO)]
    code_list = sum([[c] * (1 if n == '' else int(n)) for n, c in rle_groups], [])  # [yO yO $ yO]
    code_arr = [l.split(',') for l in ','.join(code_list).split('$')]  # [[yO yO] [yO]]
    V = [[0 if c in ['.', 'b'] else 255 if c == 'o' else ord(c) - ord('A') + 1 if len(c) == 1 else (ord(c[0]) - ord(
        'p')) * 24 + (ord(c[1]) - ord('A') + 25) for c in row if c != ''] for row in code_arr]  # [[255 255] [255]]
    maxlen = len(max(V, key=len))
    A = torch.Tensor([row + [0] * (maxlen - len(row)) for row in V]) / 255  # [[1 1] [1 0]]
    return A

if save_animals:

    animals_url = "https://raw.githubusercontent.com/Chakazul/Lenia/master/Python/animals.json"
    with contextlib.closing(urllib.request.urlopen(animals_url)) as animals_response:
        animals_data = json.loads(animals_response.read())


    for animal_data_entry in animals_data:

        if animal_data_entry["name"] == "SmoothLife":
            break

        # only entries with a params field encode an animal
        if 'params' in animal_data_entry:

            crea_filepath = f'../data/handmade_exploration/all_parameters/{animal_data_entry["name"]}.pickle'

            if os.path.exists(crea_filepath):
                continue

            #crea_parameters = torch.load("/home/mayalen/code/01-Lenia/2022.06_pnas_papers/Figure_3/resources/creaRobust/seed0/run_0000045_data_params.pickle")
            crea_parameters = {}
            crea_parameters['policy_parameters'] = {}
            crea_parameters['policy_parameters']['initialization'] = {}
            crea_parameters['policy_parameters']['initialization']['init'] = torch.zeros((40,40), dtype=torch.float32, device='cuda')

            # INIT STATE
            init_patch = torch.as_tensor(rle2arr(animal_data_entry['cells']), dtype=crea_parameters['policy_parameters']['initialization']['init'].dtype).squeeze()
            if max(init_patch.shape) > 255:
                continue

            if rescale_to_40x40:
                ## resize to (40,40)
                scaling = 40 / max(init_patch.shape)
                resized_init_size = tuple([int(s * scaling) for s in init_patch.shape])
                init_patch = Fvision.resize(init_patch.unsqueeze(0), resized_init_size,
                                            interpolation=Fvision.InterpolationMode.BILINEAR, antialias=True).squeeze(0)
            else:
                scaling = 1.0

            if init_hflip:
                init_patch = Fvision.hflip(init_patch.unsqueeze(0)).squeeze(0)

            crea_parameters['policy_parameters']['initialization']['init'] = init_patch.squeeze()

            # UPDATE RULE
            nb_k = 1
            crea_parameters['policy_parameters']['update_rule'] = {}
            crea_parameters['policy_parameters']['update_rule']['R'] = torch.as_tensor([animal_data_entry['params']['R']-15], dtype=torch.int64, device='cuda')
            crea_parameters['policy_parameters']['update_rule']['T'] = torch.as_tensor([animal_data_entry['params']['T']], dtype=torch.float32, device='cuda')
            crea_parameters['policy_parameters']['update_rule']['r'] = scaling * torch.ones(nb_k, dtype=torch.float32, device='cuda')
            if 'h' in animal_data_entry['params']:
                crea_parameters['policy_parameters']['update_rule']['h'] = torch.as_tensor([animal_data_entry['params']['h']], dtype=torch.float32, device='cuda')
            else:
                crea_parameters['policy_parameters']['update_rule']['h'] = torch.ones(nb_k, dtype=torch.float32, device='cuda')
            crea_parameters['policy_parameters']['update_rule']['b'] = -1 * torch.ones((nb_k, 5), dtype=torch.float32, device='cuda')
            b = torch.as_tensor([float(fractions.Fraction(st)) for st in animal_data_entry['params']['b'].split(',')], dtype=torch.float32, device='cuda')
            crea_parameters['policy_parameters']['update_rule']['b'][0, :len(b)] = b
            crea_parameters['policy_parameters']['update_rule']['m'] = torch.as_tensor([animal_data_entry['params']['m']], dtype=torch.float32, device='cuda')
            crea_parameters['policy_parameters']['update_rule']['s'] = torch.as_tensor([animal_data_entry['params']['s']], dtype=torch.float32, device='cuda')
            crea_parameters['policy_parameters']['update_rule']['c0'] = torch.zeros(nb_k, dtype=torch.int64, device='cpu')
            crea_parameters['policy_parameters']['update_rule']['c1'] = torch.zeros(nb_k, dtype=torch.int64, device='cpu')
            crea_parameters['policy_parameters']['update_rule']['kn'] = torch.as_tensor([animal_data_entry['params']['kn']], dtype=torch.int64, device='cpu')
            crea_parameters['policy_parameters']['update_rule']['gn'] = torch.as_tensor([animal_data_entry['params']['gn']], dtype=torch.int64, device='cpu')

            if not (crea_parameters['policy_parameters']['update_rule']['gn'] == 1).all():
                print(crea_filepath, crea_parameters['policy_parameters']['update_rule']['gn'])
            torch.save(crea_parameters, crea_filepath)


    entry_idx = -1
    for json_idx in [212, 213, 221, 222, 223, 231, "231b", 232, 233, "233b", "233c", "233d", "233s", 241, 242, 243, 251, 261, 271, "312b", 313, 322, "322b", "322c", 331, 332, 333]: #233p, 312 not working

        animals_url = f"https://raw.githubusercontent.com/Chakazul/Lenia/master/Python/found/{json_idx}.json"
        with contextlib.closing(urllib.request.urlopen(animals_url)) as animals_response:
            st = animals_response.read().decode("utf-8")
            st = "[" + st.rstrip(', \n\r\t') + "]"
            animals_data = json.loads(st)


        for animal_data_entry in animals_data:

            # only entries with a params field encode an animal
            if 'params' in animal_data_entry:

                entry_idx += 1
                crea_filepath = f'parameters/json_{json_idx}_entry_{entry_idx}.pickle'
                if os.path.exists(crea_filepath):
                    continue

                #crea_parameters = torch.load("/home/mayalen/code/01-Lenia/2022.06_pnas_papers/Figure_3/resources/creaRobust/seed0/run_0000045_data_params.pickle")
                crea_parameters = {}
                crea_parameters['policy_parameters'] = {}
                crea_parameters['policy_parameters']['initialization'] = {}
                crea_parameters['policy_parameters']['initialization']['init'] = torch.zeros((40, 40), dtype=torch.float32, device='cuda')

                # INIT STATE
                cells = animal_data_entry['cells']
                if type(cells) == str:
                    init_patch = torch.as_tensor(rle2arr(cells),
                                                 dtype=crea_parameters['policy_parameters']['initialization']['init'].dtype).squeeze()

                    if max(init_patch.shape) > 255:
                        continue

                    if rescale_to_40x40:
                        ## resize to (40,40)
                        scaling = 40 / max(init_patch.shape)
                        resized_init_size = tuple([int(s * scaling) for s in init_patch.shape])
                        init_patch = Fvision.resize(init_patch.unsqueeze(0), resized_init_size,
                                                    interpolation=Fvision.InterpolationMode.BILINEAR,
                                                    antialias=True).squeeze(0)
                    else:
                        scaling = 1.0

                    if init_hflip:
                        init_patch = Fvision.hflip(init_patch.unsqueeze(0)).squeeze(0)

                elif type(cells) == list:
                    nb_c = len(cells)

                    if nb_c > 1 or nb_c == 0:
                        continue


                    init_patch = []
                    should_continue = False
                    for channel_cells in cells:
                        channel_init_patch = torch.as_tensor(rle2arr(channel_cells),
                                                     dtype=crea_parameters['policy_parameters']['initialization'][
                                                         'init'].dtype).squeeze()

                        if max(channel_init_patch.shape) > 255:
                            should_continue = True
                            break

                        if rescale_to_40x40:
                            ## resize to (40,40)
                            scaling = 40 / max(channel_init_patch.shape)
                            resized_init_size = tuple([int(s * scaling) for s in channel_init_patch.shape])
                            channel_init_patch = Fvision.resize(channel_init_patch.unsqueeze(0), resized_init_size,
                                                        interpolation=Fvision.InterpolationMode.BILINEAR,
                                                        antialias=True).squeeze(0)
                        else:
                            scaling = 1.0

                        if init_hflip:
                            channel_init_patch = Fvision.hflip(channel_init_patch.unsqueeze(0)).squeeze(0)

                        init_patch.append(channel_init_patch.squeeze())
                    if should_continue:
                        continue

                    init_patch = torch.stack(init_patch).squeeze()

                else:
                    raise ValueError


                crea_parameters['policy_parameters']['initialization']['init'] = init_patch

                # UPDATE RULE
                if type(animal_data_entry["params"]) == dict:
                    nb_k = 1
                    animal_data_entry['params'] = [animal_data_entry['params']]


                elif type(animal_data_entry["params"]) == list:
                    nb_k = len(animal_data_entry["params"])

                else:
                    raise ValueError

                # UPDATE RULE
                crea_parameters['policy_parameters']['update_rule'] = {}
                crea_parameters['policy_parameters']['update_rule']['R'] = torch.as_tensor([animal_data_entry['params'][0]['R'] - 15], dtype=torch.int64, device='cuda')
                crea_parameters['policy_parameters']['update_rule']['T'] = torch.as_tensor([animal_data_entry['params'][0]['T']], dtype=torch.float32, device='cuda')
                crea_parameters['policy_parameters']['update_rule']['r'] = scaling * torch.ones(nb_k, dtype=torch.float32, device='cuda')
                crea_parameters['policy_parameters']['update_rule']['h'] = torch.ones(nb_k, dtype=torch.float32, device='cuda')
                crea_parameters['policy_parameters']['update_rule']['b'] = -1 * torch.ones((nb_k, 5), dtype=torch.float32, device='cuda')
                crea_parameters['policy_parameters']['update_rule']['m'] = torch.ones(nb_k, dtype=torch.float32, device='cuda')
                crea_parameters['policy_parameters']['update_rule']['s'] = torch.ones(nb_k, dtype=torch.float32, device='cuda')
                crea_parameters['policy_parameters']['update_rule']['c0'] = torch.zeros(nb_k, dtype=torch.int64, device='cpu')
                crea_parameters['policy_parameters']['update_rule']['c1'] = torch.zeros(nb_k, dtype=torch.int64, device='cpu')
                crea_parameters['policy_parameters']['update_rule']['kn'] = torch.zeros(nb_k, dtype=torch.int64, device='cpu')
                crea_parameters['policy_parameters']['update_rule']['gn'] = torch.zeros(nb_k, dtype=torch.int64, device='cpu')

                for kernel in range(nb_k):

                    if 'r' in animal_data_entry['params'][kernel]:
                        crea_parameters['policy_parameters']['update_rule']['r'][kernel] = animal_data_entry['params'][kernel]['r']
                    if 'h' in animal_data_entry['params'][kernel]:
                        crea_parameters['policy_parameters']['update_rule']['h'][kernel] = animal_data_entry['params'][kernel]['h']

                    b = torch.as_tensor([float(fractions.Fraction(st)) for st in animal_data_entry['params'][kernel]['b'].split(',')], dtype=torch.float32, device='cuda')
                    crea_parameters['policy_parameters']['update_rule']['b'][kernel, :len(b)] = b
                    crea_parameters['policy_parameters']['update_rule']['m'][kernel] = animal_data_entry['params'][kernel]['m']
                    crea_parameters['policy_parameters']['update_rule']['s'][kernel] = animal_data_entry['params'][kernel]['s']


                    if 'c' in animal_data_entry['params'][kernel]:
                        crea_parameters['policy_parameters']['update_rule']['c0'][kernel] = animal_data_entry['params'][kernel]['c'][0]
                        crea_parameters['policy_parameters']['update_rule']['c1'][kernel] = animal_data_entry['params'][kernel]['c'][1]

                    crea_parameters['policy_parameters']['update_rule']['kn'][kernel] = animal_data_entry['params'][kernel]['kn']
                    crea_parameters['policy_parameters']['update_rule']['gn'][kernel] = animal_data_entry['params'][kernel]['gn']

                if not (crea_parameters['policy_parameters']['update_rule']['gn'] == 1).all():
                    print(crea_filepath, crea_parameters['policy_parameters']['update_rule']['gn'])
                torch.save(crea_parameters, crea_filepath)

if prefilter:
    system = TmpLenia(logger=None, final_step=final_step, nb_c=1,
                      nb_k=10, wall_c=False,
                      size="({},{})".format(256, 256))

    for crea_filename in os.listdir("../data/handmade_exploration/all_parameters"):
        if crea_filename[-7:] == ".pickle":
            crea_filepath = "../data/handmade_exploration/all_parameters/" + crea_filename
            valid_crea_filepath = "../data/handmade_exploration/prefilter_parameters/" + crea_filename
            non_valid_crea_filepath = "../data/handmade_exploration/all_parameters/non_valid_after_prefilter/" + crea_filename

            system.reset_from_crea_filepath(crea_filepath, init_hflip=True, init_hoffset=10)
            system.run()
            final_mass = system._observations.states[-1, 0].cpu().numpy().sum()

            if final_mass > low_mass_threshold and final_mass < high_mass_threshold:
                is_valid = True
            else:
                print("NON VALID MASS", crea_filepath, final_mass)
                is_valid = False


            if is_valid:
                shutil.move(crea_filepath, valid_crea_filepath)
            else:
                shutil.move(crea_filepath, non_valid_crea_filepath)
