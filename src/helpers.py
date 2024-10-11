

def print_num_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}")

def print_all_model_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    dir() -> returns a list of attributes and functions
    """
    # Print the config to see the number of heads (attention)
    # Print the model to see where the attention heads are, then iterate through there parameters with below example

    if not hasattr(model, 'base_model'): #if "base_model" in dir(model): # dir() -> returns a list of attributes and functions
        for name, param in model.base_model.named_parameters():
            # print(f"Parameter Name: {name}, Parameter Value(s): {param}")
            print(f"Parameter Name: {name}")

    # # Print the attention blocks and their weight parameters in the Encoder
    # for i in range(model.config.n_layer):
    #     print(f"Attention Block {i + 1} - Number of heads: {model.config.n_head}")
    #     layer = model.base_model.decoder.layers[i]
    #     for property in dir(layer):
    #         if "attn" or "attention" in property:
    #             property_obj = getattr(layer, property)
    #             print(property_obj)

def print_model_modules(model):
    """
    Prints the model's name and list of modules.
    These modules should be traversed to find the "blocks" that are attention heads
    and either have the name "head", "h", "attn" or "attention" in them to then find their parameters
    """
    # Step 1
    # Gather the model's attributes dictionary
    model_attributes = vars(model)

    if 'name_or_path' in model_attributes:
        print(f"Model name: {model.name_or_path}")

    # Check if the "_modules" property exists within the model's attributes
    if '_modules' in model_attributes:
        # Retrieve the "_modules" property
        model_modules = model_attributes['_modules']

        # Traverse through the modules to unveil their names and types
        for name, module in model_modules.items():
            object_type = type(module)
            # Print the module names - one of these should be the same as base_model**
            print(f"Module Name: {name}, Module Type: {object_type}")
    else:
        # see if base_model exists if not _modules, but it should
        if not hasattr(model, 'base_model'): #if "base_model" in dir(model):
            for name, param in dir(model.base_model):
                print(f"Module Name: {name}")

    print("Modules found from _modules, make sure to call model.module")

def print_module_blocks(model, moduleStr: str):
    """
    Print the blocks within a module within the model.basel_model, ex. what blocks are in decoder module
    # Ex. see if block "attention" exists in decoder module, after decoder module was found in print_model_modules()
    # Ex. If print_model_modules() outputs 'transformer', that might be the base_model, so it is an attribute of model
    # In other words, the string names from '_modules' from the vars(model) output correspond to the attributes of model
    # If 'transformer' is in '_modules', then find the blocks in model.transformer**
    """
    # Step 2
    if hasattr(model, moduleStr): #if "base_model" in dir(model):
        module = getattr(model, moduleStr)
        if getattr(module, "_modules"):
            blocks = {name : getattr(module, name) for name in module._modules}

        print(blocks)
    # couldn't find the module, so just print base_model (should be the same as primary module spit out by)
    elif hasattr(model, 'base_model'):
        print(model.base_model)
    else:
        print(model)

def print_model_attributes(model):
    """
    Print the model's attributes two ways - one through config, another using vars()
    Traverse through the dictionary to seek the presence of the string "head"
    var() -> returns a dictionary of the attributes and functions
    """
    # Step 1a
    if hasattr(model, 'config_dict'):
        for key, value in model.config_dict.items():
            print(f"Object Key: {key}, Object: {value}")

    # Gather the model's attributes
    model_attributes = vars(model) # vars() -> returns a dictionary of the attributes and functions

    # Traverse through the attributes to unveil the names and types of all objects
    for name, attribute in model_attributes.items():
        object_type = type(attribute)
        print(f"Object Name: {name}, Object Type: {object_type}")

def print_named_parameters(block_object):
    """
    Prints the parameters in a specific block in a model.
    Do this after discovering module block of interest by calling print_base_model_blocks
    See if it has "attention" or "head" in its name first
    """
    # Step 4
    # Block object is a layer object from a model (ex. head, or attention block)
    generator_obj = block_object.named_parameters()
    # Generator Module.named_parameters object
    # Get list of all parameters in the attention module
    parameter_names = [param[0] for param in generator_obj]

    # Print the extracted parameter names
    print(parameter_names)