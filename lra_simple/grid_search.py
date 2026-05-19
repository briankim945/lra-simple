import itertools
from typing import Dict, List, Any


def grid_search(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Generate all combinations of hyperparameters from a grid.
    
    Args:
        param_grid: Dictionary mapping parameter names to lists of values
        
    Returns:
        List of all parameter combinations as dictionaries
    """
    keys = param_grid.keys()
    values = param_grid.values()
    
    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))
    
    return combinations


def grid_search_with_conditionals(
    base_grid: Dict[str, List[Any]],
    conditional_grids: Dict[str, Dict[str, Dict[str, List[Any]]]] = None
) -> List[Dict[str, Any]]:
    """
    Generate grid search combinations with conditional parameters.
    
    This avoids duplicating configurations when a parameter only applies
    to certain values of another parameter (e.g., momentum only for SGD).
    
    Args:
        base_grid: Dictionary of parameters that always apply
        conditional_grids: Nested dict specifying conditional parameters.
            Format: {parent_param: {parent_value: {child_param: [values]}}}
            
    Returns:
        List of all valid parameter combinations
    
    Example:
        base_grid = {"lr": [1e-3, 1e-4], "optimizer": ["adam", "sgd"]}
        conditional_grids = {
            "optimizer": {
                "sgd": {"momentum": [0.9, 0.99]}
            }
        }
        # This produces:
        # - lr=1e-3, optimizer=adam (no momentum)
        # - lr=1e-4, optimizer=adam (no momentum)
        # - lr=1e-3, optimizer=sgd, momentum=0.9
        # - lr=1e-3, optimizer=sgd, momentum=0.99
        # - lr=1e-4, optimizer=sgd, momentum=0.9
        # - lr=1e-4, optimizer=sgd, momentum=0.99
    """
    if conditional_grids is None:
        conditional_grids = {}
    
    # Generate base combinations
    base_configs = grid_search(base_grid)
    
    # Expand each base config with conditional parameters
    final_configs = []
    for config in base_configs:
        # Check if any conditional grids apply to this config
        conditional_params = {}
        for parent_param, value_map in conditional_grids.items():
            parent_value = config.get(parent_param)
            if parent_value in value_map:
                conditional_params.update(value_map[parent_value])
        
        if conditional_params:
            # Expand with conditional parameter combinations
            conditional_combos = grid_search(conditional_params)
            for conditional_config in conditional_combos:
                final_configs.append({**config, **conditional_config})
        else:
            # No conditional parameters apply
            final_configs.append(config)
    
    return final_configs