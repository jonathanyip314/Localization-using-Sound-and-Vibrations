import os
import re
import sys
import numpy as np
import math
import statistics
import pickle
import threading

from matplotlib import pyplot as plt


def scale_to_sec(time):
    return [x / 1000000 for x in time]


def decode_aud_esp(filename, data_rate):
    data_len = 33280
    frame_len = 1030

    with open(filename, 'rb') as f:
        raw_stream = f.read()

    # removing redundant bytes
    stream = b''
    pattern = raw_stream[:20]
    prev_ind = 20
    for pat in re.finditer(pattern, raw_stream):
        if pat.start() > 20:
            stream += raw_stream[prev_ind: prev_ind + ((pat.start() - prev_ind) // data_len) * data_len]
            prev_ind = pat.end()
    stream += raw_stream[prev_ind:]
    del raw_stream

    samples_len = (len(stream) // 33280) * 8000
    output = np.zeros((2, samples_len), np.double)
    missing_frames = 0

    prev_fr_no = 0
    count = 0
    processed_frames = 0
    duplicates = 0
    base = 0

    # setting base-data value for each device
    if '_01_' in filename:
        base = 225
    elif '_02_' in filename:
        base = 229
    elif '_03_' in filename:
        base = 229
    else:
        print("Houston we have a problem ... ?")
        # exit()

    for i in range(0, len(stream) - data_len, data_len):

        count = 0
        processed_frames += 1
        # tracking missing time
        if i > 0 and (stream[i + data_len - 1] - prev_fr_no > 1 or stream[i + data_len - 1] - prev_fr_no + 256 > 1):
            missing_frames += (stream[i + data_len - 1] - prev_fr_no)

            if stream[i + data_len - 1] - prev_fr_no == 0:
                duplicates += 1
                processed_frames -= 1
                continue

        elif i > 0 and (stream[i + data_len - 1] - prev_fr_no < 1 and stream[i + data_len - 1] - prev_fr_no > -254):
            print(stream[i + data_len - 1] - prev_fr_no)
            continue

        for j in range(i, i + data_len - frame_len, frame_len):

            # getting base timestamp
            base_data = int.from_bytes(stream[j:j + 2][::-1], 'big')
            base_hi = int.from_bytes(stream[j + 2:j + 4][::-1] + stream[j + 4:j + 6][::-1], 'big')
            base_lo = int.from_bytes(stream[j + 6:j + 8][::-1] + stream[j + 8:j + 10][::-1], 'big')
            if base_hi > 1:
                continue
            # if i < 5000000:
            # print(stream[i+data_len-1], i//data_len, i, j//frame_len, j, base_data, base_hi, base_lo)
            try:
                base_time = base_hi * (2 ** 32) + base_lo
            except Exception as ex:
                print(stream[i + data_len - 1], i // data_len, j // frame_len, base_data, base_hi, base_lo)
                missing_frames += 1
                base_time = 0
                exit()
                continue

            output[0, (processed_frames - 1) * 8000 + count] = base_data
            output[1, (processed_frames - 1) * 8000 + count] = base_time
            count += 1

            for k in range(j + 10, j + frame_len, 4):

                # only 8000 valid samples in space of 8192
                if count >= 8000:
                    # print (i, j, k, count)
                    break

                data_point = int.from_bytes(stream[k:k + 2][::-1], 'big')
                # filtering data-point
                if data_point > base + 5 and data_point <= base + 15:
                    data_point -= 10
                elif data_point > base + 15 and data_point <= base + 25:
                    data_point -= 20
                elif data_point > base + 25 and data_point <= base + 35:
                    data_point -= 30
                elif data_point > base + 35:
                    data_point -= 40
                elif data_point < base - 5 and data_point >= base - 15:
                    data_point += 10
                elif data_point < base - 15 and data_point >= base - 25:
                    data_point += 20
                elif data_point < base - 25 and data_point >= base - 35:
                    data_point += 30
                elif data_point < base - 35:
                    data_point += 40

                off_time = int.from_bytes(stream[k + 2:k + 4][::-1], 'big', signed=True)

                output[0, (processed_frames - 1) * 8000 + count] = data_point
                output[1, (processed_frames - 1) * 8000 + count] = base_time - off_time
                count += 1

        prev_fr_no = stream[i + data_len - 1]
        pre_base = base_time
    # print("{} duplicates and missed {} frames".format(duplicates, missing_frames))

    # cache data for later use
    '''f = open(os.path.join(cache_path, bytes(cached)), 'wb')
    pickle.dump(output, f)'''

    return output[:, 10:output.shape[1] - duplicates * data_rate].copy()


def decode_imu_esp(filename, data_rate):
    data_len = 8001
    frame_len = 20
    period = 1000000 / data_rate  # in micro-seconds

    with open(filename, 'rb') as f:
        stream = f.read()[20:]

    samples_len = (len(stream) // data_len) * 400
    output = np.zeros((4, samples_len), np.double)
    # missing_frames = 0

    # prev_fr_no = 0
    count = 0
    # processed_frames = 0
    # duplicates = 0
    i = 0
    faults = 0

    for i in range(0, len(stream), data_len):

        if i + data_len - 1 >= len(stream):
            break

        for j in range(i, i + data_len - frame_len, frame_len):

            ind = j // frame_len - j // (frame_len * data_len)

            # axis data
            data_x = int.from_bytes(stream[j:j + 2][::-1], 'big', signed=True) / (32768 / 8)
            data_y = int.from_bytes(stream[j + 2:j + 4][::-1], 'big', signed=True) / (32768 / 8)
            data_z = int.from_bytes(stream[j + 4:j + 6][::-1], 'big', signed=True) / (32768 / 8)

            # timestamps
            time_hi = int.from_bytes(stream[j + 14:j + 16][::-1] + stream[j + 12:j + 14][::-1], 'big')
            time_lo = int.from_bytes(stream[j + 16:j + 18][::-1] + stream[j + 18:j + 20][::-1], 'big')

            if (time_hi == 0 and time_lo == 0) or data_z >= -0.01 and ind > 0:
                output[0, ind] = output[0, ind - 1]
                output[1, ind] = output[1, ind - 1]
                output[2, ind] = output[2, ind - 1]
                output[-1, ind] = output[-1, ind - 1] + period
                count += 1

            elif not (time_hi != 0 and time_lo != 0) and time_hi < 2:
                output[0, ind] = data_x
                output[1, ind] = data_y
                output[2, ind] = data_z
                output[-1, ind] = time_hi * (2 ** 32) + time_lo
                count += 1

            else:
                faults += 1

    return output


def timelimiter(low, high, d_array):  # this function creates a time range for searching an event
    first_round_flag = True
    low_index = None
    high_index = None
    for x in enumerate(d_array):
        if x[1] < low:
            continue
        if first_round_flag:
            first_round_flag = False
            low_index = x[0]
        if x[1] > high:
            high_index = x[0]
            break
    return low_index, high_index


def micEventLocator(
        d_array):  # This function detects that an event has occured at microphones and returns a time range in which the event has occured
    events = 0
    x = 0
    timearr = []
    while x < len(d_array[0, :]):
        if d_array[0, x] > 2700:
            arr = d_array[0, x:x + 100]
            timestamps = d_array[-1, x:x + 100]
            maxi = max(arr)
            index = np.where(arr == maxi)[0][0]
            # print("Sensor MAX: ", maxi)
            if timestamps[index] / 1000000 > 50 and timestamps[index] / 1000000 < 300:
                t = math.floor(timestamps[index] / 1000000)
                timearr.append((t - 2, t + 2))
            x = x + 5000
        x += 1
    return (timearr)


def findTimes(d_array, timeRange, threshold):  # return the timestamp at which the event occured at the microphone
    lower = timeRange[0]
    upper = timeRange[1]
    low_index, high_index = timelimiter(lower, upper, scale_to_sec(d_array[-1, :]))
    x = d_array[0, low_index:high_index]
    y = d_array[-1, low_index:high_index]
    try:
        maxi = max(x)
        # print (maxi)
        if maxi < threshold or maxi > 4500:
            return None
        index = np.where(x == maxi)[0][0]
        return (y[index] / 1000000)
    except:
        return None


def pythag(x, y, x1, y1):  # calculate the distance from a point to another
    distancex = x1 - x
    distancey = y1 - y
    distance = math.sqrt(distancex ** 2 + distancey ** 2)
    return distance


# how timeDiffArr is created. Commented out because it is saved and dont want to compile everytime
# timeDiffArr = []
# for i in np.arange(0, 1, .1):
#     timeDiff = []
#     for j in np.arange(0, 1, .1):
#         t1 = pythag(0, 0, j, i)
#         t2 = pythag(1, 0, j, i)
#         t3 = pythag(1, 1, j, i)
#
#         diff12 = (t1 - t2) / 343
#         diff13 = (t1 - t3) / 343
#         diff23 = (t2 - t3) / 343
#         timeDiff.append([diff12, diff13, diff23])
#     timeDiffArr.append(timeDiff)
#
#
# with open('timeDiffPickle','wb') as file:
#     pickle.dump(timeDiffArr,file)
# file.close()

# grab the timeDiffArray that is saved
with open('timeDiffPickle', 'rb') as file:
    timeDiffArr = pickle.load(file)


# print(timeDiffArr)


def AllAud(a, b, c):  # localization based on 3 microphones
    p = a - b
    q = a - c
    r = b - c
    LowestErr = 999

    for i in range(0, 9):
        for j in range(0, 9):
            err = abs(timeDiffArr[j][i][0] - p)
            err2 = abs(timeDiffArr[j][i][1] - q)
            err3 = abs(timeDiffArr[j][i][2] - r)
            AvgErr = (err + err2 + err3) / 3
            if AvgErr < LowestErr:
                LowestErr = AvgErr
                guess = [j, i]
    # print(LowestErr)
    return guess


def TwoAud(Sensors):  # localization based on 2 microphones
    p = Sensors[0][1] - Sensors[1][1]
    if Sensors[0][0] == "S1" and Sensors[1][0] == "S2":
        k = 0
    if Sensors[0][0] == "S1" and Sensors[1][0] == "S3":
        k = 1
    if Sensors[0][0] == "S2" and Sensors[1][0] == "S3":
        k = 2
    lowestErr = 999
    for i in range(0, 9):
        for j in range(0, 9):
            err = abs(timeDiffArr[j][i][k] - p)
            if err < lowestErr:
                lowestErr = err
                guess = [j, i]
    # print (lowestErr)
    return guess


# trained on all but last 5 events
# 10 locations
# format: {imu_mean}, {imu_stdv}
locations = [[2, 2], [3, 3], [3, 7], [4, 6], [5, 4], [5, 7], [6, 4], [7, 3], [7, 7], [8, 8]]
imu_mean_std_trained_data = [(0.3035481770833333, 2.437252790190614), (-0.7532784598214286, 1.0408444663530396),
                             (-0.185595703125, 0.36990816314037245), (None, None),
                             (-2.4500558035714284, 2.608362907138707),
                             (None, None), (-1.8760986328125, 0.964585829636136),
                             (-1.2797154017857142, 0.556574333915539),
                             (-1.19599609375, 0.9955847862281798), (None, None)]


def imuScore(ratio):
    result = []
    for mean, stdv in imu_mean_std_trained_data:
        if mean is None or stdv is None:
            result.append(None)
            continue
        edge_value = 2 * stdv
        if ratio >= mean + 2 * stdv or ratio <= mean - 2 * stdv:
            result.append(0.0)
        elif ratio == mean:
            result.append(1.0)
        elif ratio / mean > 0:
            result.append(1 - abs((ratio - mean) / edge_value))
        else:
            result.append(1 - abs((ratio + mean) / edge_value))

    return result


def Allimu(event_list):
    arr = []
    for event_range in event_list:
        amp1 = ampIMU(data4, event_range)
        amp2 = ampIMU(data5, event_range)

        if amp1 == None or amp2 == None:
            # print(f'score @{event_range} for {dir}: {None}')
            arr.append(None)
        else:
            # print(f'score @{event_range} for {dir}: {imuScore(amp1 - amp2)}')
            arr.append(imuScore(amp1 - amp2))
    return arr


def mergeEvents(eventArray):
    Events = eventArray[:]  # shallow copies the list
    for index, event in enumerate(Events):
        if index != 0 and Events[index - 1][1] > event[0]:
            Events[index - 1] = (Events[index - 1][0], event[1])
            del Events[index]
    return Events


def ampIMU(d_array, timeRange):
    lower = timeRange[0]
    upper = timeRange[1]
    low_index, high_index = timelimiter(lower, upper, scale_to_sec(d_array[-1, :]))
    if low_index is None or high_index is None:
        return None
    AMP = d_array[1, low_index:high_index + 1]
    # Time = d_array[-1, low_index:high_index]
    maxi = max(AMP)
    mini = min(AMP)
    height = abs(maxi - mini)
    if height < .1:
        return None
    return (height)


def score_to_loc(event_score):
    index = event_score.index(max(event_score))
    if index is None:
        return None

    return locations[index]


def computation_thread(data, data2, data3, data4, data5):
    print("real event occured at:", dir)

    print("finding events")
    Events = sorted(set(micEventLocator(data) + micEventLocator(data2) + micEventLocator(
        data3)))
    Events = mergeEvents(Events)
    print(Events[:5])
    input("Press ENTER to continue")
    # print (Events)

    print("Performing Localization using IMU...")
    filtered_events = [None if event is None else [i for i in event if i is not None] for event in Allimu(Events[:5])]
    imu_guess = [None if event is None else score_to_loc(event) for event in filtered_events]

    print("IMU GUESS: ", imu_guess)
    event = 0
    aud_guess = []
    input("Press ENTER to continue")
    print("Performing localization using Microphone...")
    for i in Events[:5]:

        AudCount = []
        T1 = findTimes(data, i, 3000)
        if T1 is not None:
            AudCount.append(("S1", T1))
        T2 = findTimes(data2, i, 3000)
        if T2 is not None:
            AudCount.append(("S2", T2))
        T3 = findTimes(data3, i, 3000)
        if T3 is not None:
            AudCount.append(("S3", T3))
        if len(AudCount) == 3:
            # print("Event",event,"(All AUD): ",AllAud(T1,T2,T3))
            aud_guess.append(AllAud(T1, T2, T3) + [3])
        elif len(AudCount) == 2:
            aud_guess.append(TwoAud(AudCount) + [2])
            # print("Event",event,"(2AUD): ",TwoAud(AudCount))
        else:
            # print("Event",event,"<1 Aud")
            aud_guess.append(None)
        event += 1

    print("MIC GUESS: ", aud_guess)

    truevalue = [3, 3]  # UPDATE THIS GROUND TRUTH VALUE TO CALCULATE THE ERROR!
    error = []
    for x in range(0, len(aud_guess)):
        input("Press ENTER to continue")
        print("----------Event ", x, "----------")
        # print("GUESSES:",aud_guess[x],imu_guess[x])
        if aud_guess[x] is None and imu_guess[x] is None:
            FinalXGuess = None
            FinalYGuess = None
        elif aud_guess[x] is None:  # if there is no estimation from the microphones, use the imu estimation
            FinalXGuess = imu_guess[x][0]
            FinalYGuess = imu_guess[x][1]
        elif imu_guess[x] is None:  # if there is no estimation from the imus, use the microphone estimation
            FinalXGuess = aud_guess[x][0]
            FinalYGuess = aud_guess[x][1]
        else:
            if aud_guess[x][2] == 2:  # if 2 microphones detected the event, weights are 50-50
                FinalXGuess = (aud_guess[x][0] + imu_guess[x][0]) / 2
                FinalYGuess = (aud_guess[x][1] + imu_guess[x][1]) / 2
            if aud_guess[x][2] == 3:  # if 3 microphones detected the event, weights are 80-20
                FinalXGuess = .8 * aud_guess[x][0] + .2 * imu_guess[x][0]
                FinalYGuess = .8 * aud_guess[x][1] + .2 * imu_guess[x][1]
        print("Combined Guess: ", [FinalXGuess, FinalYGuess])

        if FinalXGuess is not None:
            print("Error:", pythag(FinalXGuess, FinalYGuess, truevalue[0], truevalue[1]))
            error.append(pythag(FinalXGuess, FinalYGuess, truevalue[0], truevalue[1]))
        else:
            print("NO GUESS")

    print("\n\nAVG ERROR: ", statistics.mean(error))


if __name__ == "__main__":

    data_rate_mic = 8000
    data_rate_imu = 200
    count = 0
    dir = '3x3'  # PUT THE NAME OF THE DIRECTORY CONTAINING ALL OF THE DATA FILES HERE!
    for filename in os.listdir(dir):
        if count == 0:
            data = decode_aud_esp(dir + '/' + filename, 8000)
        if count == 1:
            data2 = decode_aud_esp(dir + '/' + filename, 8000)
        if count == 2:
            data3 = decode_aud_esp(dir + '/' + filename, 8000)
        if count == 3:
            data4 = decode_imu_esp(dir + '/' + filename, 200)
        if count == 4:
            data5 = decode_imu_esp(dir + '/' + filename, 200)
        count += 1

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)
    ax1.plot(scale_to_sec(data[-1, :]), data[0, :])
    ax1.set_xlim(50, 300)
    ax1.set_ylabel('Amplitude')
    ax2.plot(scale_to_sec(data2[-1, :]), data2[0, :])
    ax2.set_xlim(50, 300)
    ax2.set_ylabel('Amplitude')
    ax3.plot(scale_to_sec(data3[-1, :]), data3[0, :])
    ax3.set_xlim(50, 300)
    ax3.set_ylabel('Amplitude')
    ax4.plot(scale_to_sec(data4[-1, :]), data4[1, :])
    ax4.set_xlim(50, 300)
    ax4.set_ylim(-5, 5)
    ax4.set_ylabel('Amplitude')
    ax5.plot(scale_to_sec(data5[-1, :]), data5[1, :])
    ax5.set_xlim(50, 300)
    ax5.set_ylim(-5, 5)
    ax1.set_title('Microphone and IMU Sensor Data ')
    ax5.set_ylabel('Amplitude')
    ax5.set_xlabel('time(seconds)')
    compute = threading.Thread(target=computation_thread, args=(data, data2, data3, data4, data5))
    compute.start()
    plt.show()
