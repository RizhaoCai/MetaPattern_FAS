import re

def parse_attr_from_video_name(video_name):
    if 'CASIA' in video_name:
        return parse_casia_attr(video_name)

    if 'REPLAY-ATTACK' in video_name.upper():
        return parse_replay_attr(video_name)

    if 'MSU' in video_name.upper():
        return parse_msu_attr(video_name)

    if 'OULU' in video_name.upper():
        return parse_oulu_attr(video_name)


def parse_casia_attr(video_name):
    """

    :param video_name: example: 6_1_2_3
    :return:
    """

    video_name = re.findall('(\d|HR_\d).avi', video_name)[0]
    env = 'casia_env'
    # print(video_name)
    if video_name in ['HR_1', 'HR_2', 'HR_3', 'HR_4']:
        camera = 'High Resolution'
    else:
        camera = 'Low Resolution'

    if video_name in ['1', '2', 'HR_1']:
        type = 'real'
    elif video_name in ['3', '4', '5', '6', 'HR_2', 'HR_3']:
        type = 'print'
    elif video_name in ['7', '8', 'HR_4']:
        type = 'screen'

    attr = {
        'env': env,
        'camera': camera,
        'face': type,
    }

    return attr


def parse_msu_attr(video_name):
    """

    :param video_name: example: 6_1_2_3
    :return:
    """

    # Env

    env = 'msu_env'

    if 'laptop' in video_name:
        camera = 'Laptop'
    elif 'android' in video_name:
        camera = 'Andriod'

    if 'attack' not in video_name:
        type = 'real'
    elif 'printed_photo' in video_name:
        type = 'print'
    elif 'video' in video_name:
        type = 'screen'

    attr = {
        'env': env,
        'camera': camera,
        'face': type,
    }

    return attr


def parse_replay_attr(video_name):
    """

    :param video_name: example: 6_1_2_3
    :return:
    """

    # Env
    if 'adverse' in video_name:
        env = 'replay_env_adverse'
    else:
        env = 'replay_env_controlled'

    camera = 'Webcam'

    if 'attack' not in video_name:
        type = 'real'
    elif 'print' in video_name:
        type = 'print'
    else:
        type = 'screen'

    attr = {
        'env': env,
        'camera': camera,
        'face': type,
    }

    return attr


def parse_oulu_attr(video_name):
    """

    :param video_name: example: 6_1_2_3
    :return:
    """

    video_name = re.findall('\d_\d_\d+_\d', video_name)[0]
    attr = video_name.split('_')
    # print(video_name)
    env = 'oulu_env_{}'.format(attr[0])
    cams = ['Samsung Galaxy S6 edge', 'HTC Desire EYE', 'MEIZU X5', 'ASUS Zenfone Selfie', 'Sony XPERIA C5 Ultra Dual',
            'OPPO N3']
    type = ''

    camera = cams[int(attr[1])]

    if attr[3] == '1':
        type = 'real'
    elif attr[3] in ['2', '3']:
        type = 'print'
    elif attr[3] in ['4', '5']:
        type = 'screen'

    attr = {
        'env': env,
        'camera': camera,
        'face': type,
    }

    return attr