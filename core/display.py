
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"


def format_return(value: float) -> str:
    """Format return: green if positive, plain if negative."""
    text = f"{value * 100:>9.1f}%"
    if value > 0:
        return f"{GREEN}{text}{RESET}"
    return text


def format_drawdown(value: float) -> str:
    """Format drawdown: red if worse than -40%, plain otherwise."""
    text = f"{value * 100:>9.1f}%"
    if value < -0.40:
        return f"{RED}{text}{RESET}"
    return text


def print_row(label: str, total_return: float, max_dd: float,
              col_width: int = 14) -> None:
    """Print one result row with conditional colour formatting."""
    ret_str = format_return(total_return)
    dd_str = format_drawdown(max_dd)

    print(f"{label:<{col_width}}{ret_str}{dd_str}")


def print_header(col1: str = "Instrument",
                 col2: str = "Return",
                 col3: str = "Drawdown",
                 col_width: int = 14) -> None:
    """Print table header."""
    print(f"{'':>{col_width - len(col1)}}{col1}"
          f"{col2:>10}{col3:>10}")
