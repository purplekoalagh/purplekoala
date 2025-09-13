'''
Discrete module containing function for discrete derivative of a timeseries
'''
def diff(t: list, x: list) -> list:
    """
    Calculates the discrete derivative v(t) of timeseries

    Formula: v(t) = x(t_k)-x(t_k-1)/ t_k - t_k-1

    Args:
        t (list): List of time values
        x (list): List of the signal values at the corresponding time based on index place
    
    Returns:
        v (list): A list of discrete derivatives at the time values

    Raises:
        ValueError: If t and x are not of equal length

    Examples:
        >>> diff([23.1, 22.5, 23.5, 21.88, 22.5, 23.5, 24.88, 25], [0, 0.1, 0.3, 0.4, 0.55, 0.67, 0.71, 1])
        [-6.000000000000014, 5.0, -16.200000000000003, 4.133333333333339, 8.333333333333334, 34.50000000000004, 0.41379310344827924]
    """

    if len(x) != len(t):
        raise ValueError("t and x must have equal length")
    
    v = []

    if t[0] != 0:       
        v.append((x[0]/t[0]))

    for k in range(1, len(t)):

        top = (x[k] - x[k-1])
        bottom = (t[k]-t[k-1])
        v.append(top/bottom)

    return v

if __name__ == "__main__":
    # Example Usage
    print("Discrete Derivative v(t)")
    t = [0, 0.1, 0.3, 0.4, 0.55, 0.67, 0.71]
    x = [23.1, 22.5, 23.5, 21.88, 22.5, 23.5, 24.88]

    print(f"AP: x={x}, t={t}")

    v = diff(t,x)
    print(f"v(t)={v}")


