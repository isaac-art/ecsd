from colorama import Fore, Back, Style

def log(message, color='BLACK'):
    # set print color
    if color == 'BLACK':
        print(Fore.BLACK + message)
    elif color == 'RED':
        print(Fore.RED + message)
    elif color == 'GREEN':
        print(Fore.GREEN + message)
    elif color == 'YELLOW':
        print(Fore.YELLOW + message)
    elif color == 'BLUE':
        print(Fore.BLUE + message)
    print(Style.RESET_ALL)
    