import hashlib

def generate_scenario(student_id):
    """Generates unique warehouse parameters based on Student ID."""
    # Ensure ID is an integer
    s_id = int(student_id)
    last_digit = s_id % 10
    
    # Uncertainty: Slip probability scales with student ID
    # 5% to 14% chance of failing a move action
    slip_prob = 0.05 + (last_digit * 0.01) 
    
    # Environment Scale
    grid_size = 8 if (s_id % 2 == 0) else 10
    
    return {
        "p_slip": slip_prob,
        "size": grid_size,
        "gamma": 0.95,  # Discount Factor
        "rewards": {"goal": 100, "pit": -100, "step": -1}
    }