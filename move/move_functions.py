import pyautogui
import time
import pygetwindow


WOW_WINDOW_NAME = "World of Warcraft"
THREAD_OVERLAP_TIME = 0.5
WOW_CLIENT_RESIZE = (800, 600)


def jump():
    click_wow_window()
    # pyautogui.press("space")
    # time.sleep(1.5)

    pyautogui.keyDown("space")
    time.sleep(0.9)
    pyautogui.keyUp("space")
    time.sleep(0.6)


def nova():
    pyautogui.press("2")


def move(dur, move_key, spam_button="v", spam_freq=10):
    click_wow_window()
    time.sleep(0.01)
    spams = 0
    start_time = time.time()

    if len(move_key) == 1:
        pyautogui.keyDown(move_key)
    else:
        for key in move_key:
            pyautogui.keyDown(key)

    while 1:
        time_taken = time.time() - start_time
        if time_taken > dur:
            break
        target_spams = time_taken // 10
        if target_spams > spams:
            pyautogui.press(spam_button)
            spams += 1

    if len(move_key) == 1:
        pyautogui.keyUp(move_key)
    else:
        for key in move_key:
            pyautogui.keyUp(key)


def left(dur):
    move(dur, move_key="q")


def mount():
    click_wow_window()
    pyautogui.press("f")
    time.sleep(3.5)


def right(dur):
    move(dur, move_key="e")


def forward(dur):
    move(dur, move_key="w")


def backward(dur):
    move(dur, move_key="s")


def forward_right(dur):
    move(dur, move_key=["w", "e"])


def forward_left(dur):
    move(dur, move_key=["w", "q"])


def backward_left(dur):
    move(dur, move_key=["s", "q"])


def backward_right(dur):
    move(dur, move_key=["s", "e"])


def focus_wow():
    name = WOW_WINDOW_NAME

    try:
        window = pygetwindow.getWindowsWithTitle(name)[0]
        window.activate()
    except:
        return False

    return True


def click_wow_window():
    window_pos = pygetwindow.getWindowsWithTitle(WOW_WINDOW_NAME)[0].topleft
    window_hoirzontal_80_percent = window_pos[0] + 0.8 * WOW_CLIENT_RESIZE[0]
    window_vertical_20_percent = window_pos[1] + 0.2 * WOW_CLIENT_RESIZE[1]
    coord = (window_hoirzontal_80_percent, window_vertical_20_percent)
    pyautogui.click(*coord)


def shield():
    click_wow_window()
    pyautogui.keyDown("shift")
    pyautogui.press("v")
    pyautogui.keyUp("shift")
    time.sleep(1.1)
    pyautogui.press("v")


def iceblock():
    click_wow_window()
    pyautogui.keyDown("shift")
    pyautogui.press("f9")
    pyautogui.keyUp("shift")


def arcane_intellect():
    click_wow_window()
    pyautogui.keyDown("shift")
    pyautogui.press("f10")
    pyautogui.keyUp("shift")


def turn_left(duration):
    click_wow_window()
    pyautogui.keyDown("a")
    time.sleep(duration)
    pyautogui.keyUp("a")


def turn_right(duration):
    click_wow_window()
    pyautogui.keyDown("d")
    time.sleep(duration)
    pyautogui.keyUp("d")


def orientate_wow():
    try:
        window = pygetwindow.getWindowsWithTitle(WOW_WINDOW_NAME)[0]
        screen_dims = pyautogui.size()
        left = screen_dims[0] - WOW_CLIENT_RESIZE[0] - 15
        window.moveTo(left, 0)
    except:
        pass


def blink():
    # press shift 2
    click_wow_window()
    pyautogui.keyDown("shift")
    pyautogui.press("2")
    pyautogui.keyUp("shift")


def run_jump(wait_time):
    click_wow_window()
    end_buffer_time = 0.1
    if wait_time <= end_buffer_time:
        end_buffer_time = 0
    pyautogui.keyDown("w")
    time.sleep(wait_time - end_buffer_time)
    pyautogui.press("space")
    time.sleep(end_buffer_time)
    pyautogui.keyUp("w")


def arcane_explosion():
    pyautogui.keyDown("shift")
    pyautogui.press("g")
    pyautogui.keyUp("shift")
    time.sleep(1.1)


def format_time(seconds: int):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def ratchet_to_wc():
    start_time = time.time()
    jump()
    backward(1)
    right(3)
    backward_right(1.82)
    right(5)
    forward_right(5)
    iceblock()
    forward(30)
    left(3)
    forward_left(2)
    forward(46)
    forward_right(20)
    forward(30)
    forward_right(10)
    forward(30)
    forward_right(8)
    forward(32)
    left(3)
    forward_left(3)
    forward(4)
    forward(9)
    right(4)
    forward(3)
    forward_right(3)
    shield()
    forward(0.5)
    right(4)
    forward(5)
    right(10)
    right(2)
    forward(1)
    right(2)
    forward_right(2)
    right(3)
    forward_right(1.5)
    right(3)
    right(3)
    right(0.5)
    backward_right(1)
    shield()
    backward_right(4)
    backward_right(7)
    right(2)
    forward_right(0.5)
    forward_right(0.5)
    forward_right(0.5)
    right(2)
    right(2)
    backward(1.3)
    forward_right(0.7)
    right(3)
    shield()
    forward_right(1.5)
    right(2)
    forward(1.5)
    right(3)
    right(3)
    backward_right(2)
    backward_right(4)
    right(0.5)
    backward_right(4)
    backward(6)
    for i in range(6):
        backward(1)
        jump()
    for i in range(3):
        backward(2)
        jump()
    left(1.6)
    left(0.7)
    for i in range(3):
        backward(2)
        jump()
    shield()
    left(2)
    backward(2.5)
    left(2)
    backward(2.5)
    right(1)
    backward(2.5)
    left(1.5)
    backward(4)
    right(1)
    backward(8)
    left(1)
    backward(5)
    right(1)
    backward(3)
    right(1)
    right(1)
    shield()
    backward(3)
    backward(13)
    left(3)
    backward(3)
    backward(3)
    shield()
    left(4)
    forward(0.4)
    left(4)
    forward_left(2.5)
    backward(1)
    left(2)
    forward(3)
    forward_left(1.5)
    forward(3)
    forward_left(1.5)
    forward_left(3)
    forward_left(3)
    forward(4)
    forward_left(3)
    forward_left(3)
    time_taken = format_time(int(time.time() - start_time))
    print("Ran from Ratchet to WC in: {}s".format(time_taken))


def instance_port_to_water():
    start_time = time.time()
    forward(2)
    forward_left(3)
    left(6)
    backward_left(3)
    left(1)
    backward_left(1)
    left(1)
    forward_left(2.3)
    left(1.1)
    time.sleep(1.3)
    blink()
    forward(3.8)
    backward_right(0.5)
    right(1)
    backward_right(0.5)
    time.sleep(10)
    forward_right(2)
    right(1)
    forward(0.3)
    right(2)
    blink()
    forward_right(2)
    right(3)
    backward_right(0.5)
    right(4)
    forward(0.2)
    turn_left(0.35)
    jump()
    forward(0.3)
    forward_right(0.3)
    forward(1.7)
    time.sleep(3)
    for i in range(4):
        arcane_explosion()
    time_taken = format_time(int(time.time() - start_time))
    print(f"Took {time_taken}s to run from port to fish spot")


def ratchet_inn_chair_to_wc_fish_spot():
    orientate_wow()
    time.sleep(2)
    ratchet_to_wc()
    time.sleep(10)  # wait for loading
    instance_port_to_water()


def gadgetzan_inn_to_tanaris_shore():
    orientate_wow()
    time.sleep(2)

    backward(4)
    left(4)
    backward_left(9)
    backward(0.5)
    left(6)

    shield()
    time.sleep(2)
    iceblock()
    time.sleep(2)
    mount()

    forward(9)
    forward_left(2)
    forward(4)

    forward(2)
    forward_left(3)

    forward_left(3)
    forward(3)
    forward(3)
    forward_left(3)
    forward(9)
    forward(9)

    forward_left(2)
    forward(4)
    forward(9)

    forward_left(9)
    left(2)
    forward(9)
    forward_left(9)
    left(4)
    left(4)

    left(1)
    forward_left(3)
    mount()
    left(3)
    forward_left(3)

    for _ in range(5):
        forward(1)
        jump()

    for _ in range(5):
        left(2)
        jump()

    for _ in range(5):
        forward(2)
        jump()

    for _ in range(7):
        left(2)
        jump()

    for _ in range(3):
        forward_left(2)
        jump()

    for _ in range(5):
        left(2)
        jump()

    for _ in range(5):
        left(2)
        jump()

    for _ in range(10):
        left(2)
        jump()

    for _ in range(7):
        left(2)
        jump()

    backward(5)
    for _ in range(3):
        jump()

    backward(5)
    for _ in range(3):
        jump()

    backward(4)


if __name__ == "__main__":
    pass
