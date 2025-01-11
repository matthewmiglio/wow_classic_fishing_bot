import os
import platform
import psutil
import screeninfo
import ctypes
import zipfile

def get_system_info():
    return {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "architecture": platform.architecture(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "memory_info": psutil.virtual_memory(),
        "swap_memory": psutil.swap_memory(),
        "disk_partitions": psutil.disk_partitions(),
        "disk_usage": {partition.mountpoint: get_disk_usage(partition.mountpoint) for partition in psutil.disk_partitions()},
        "network_info": psutil.net_if_addrs(),
        "boot_time": psutil.boot_time(),
        "current_directory": os.getcwd(),
        "environment_variables": os.environ,
        "display_settings": get_display_settings()
    }

def get_display_settings():
    try:
        screens = screeninfo.get_monitors()
        return [{"width": screen.width, "height": screen.height, "name": screen.name} for screen in screens]
    except ImportError:
        return "screeninfo module not installed"

def get_cpu_info():
    return {
        "physical_cores": psutil.cpu_count(logical=False),
        "total_cores": psutil.cpu_count(logical=True),
        "cpu_freq": psutil.cpu_freq(),
        "cpu_stats": psutil.cpu_stats(),
        "cpu_times": psutil.cpu_times(),
        "cpu_percent": psutil.cpu_percent(interval=1)
    }

def get_memory_info():
    return {
        "virtual_memory": psutil.virtual_memory(),
        "swap_memory": psutil.swap_memory()
    }

def get_disk_info():
    partitions = psutil.disk_partitions()
    disk_usage = {partition.mountpoint: get_disk_usage(partition.mountpoint) for partition in partitions}
    return {
        "partitions": partitions,
        "disk_usage": disk_usage
    }

def get_disk_usage(mountpoint):
    try:
        return psutil.disk_usage(mountpoint)
    except PermissionError:
        return "PermissionError: The device is not ready"

def get_network_info():
    return {
        "net_io_counters": psutil.net_io_counters(),
        "net_if_addrs": psutil.net_if_addrs(),
        "net_if_stats": psutil.net_if_stats(),
        "net_connections": psutil.net_connections()
    }

def get_current_directory():
    return os.getcwd()

def get_environment_variables():
    return os.environ

def collect_all_system_info():
    def dict_2_string(dictionary):
        string = ""
        for k, v in dictionary.items():
            this_string = f'{k}:::{v}\n'
            string += this_string
        string = string[:-1]
        return string

    big_dict = {}
    big_dict.update(get_system_info())
    big_dict.update(get_cpu_info())
    big_dict.update(get_memory_info())
    big_dict.update(get_disk_info())
    big_dict.update(get_network_info())
    big_dict["current_directory"] = get_current_directory()
    big_dict["window_scaling"] = get_window_scaling()

    for k, v in big_dict.items():
        v_len = len(str(v))
        if v_len > 100:
            big_dict[k] = f"Length: {v_len}"


    return dict_2_string(big_dict)

def get_window_scaling():
    try:
        user32 = ctypes.windll.user32
        dpi = user32.GetDpiForSystem()  # Get the DPI scaling factor for the system
        return dpi / 96 * 100  # 96 DPI is the default DPI for 100% scaling
    except Exception as e:
        return str(e)


def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)

    size_in_gb = total_size / (1024 ** 3)
    size_in_mb = total_size / (1024 ** 2)

    if size_in_gb > 0.5:
        return f"{size_in_gb:.2f} GB"
    else:
        return f"{size_in_mb:.2f} MB"





if __name__ == "__main__":
    all_system_info = collect_all_system_info()
    print(all_system_info)
