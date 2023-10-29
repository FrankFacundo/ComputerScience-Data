"""
author: Frank Facundo
"""

import asyncio
import csv
import os
import re
import time

from datetime import datetime

from kasa import SmartPlug

class EnergyTracker:
    """
    A class that tracks energy usage and saves data to a CSV file.
    """
    IP_DEVICE = os.environ['ENERGY_TRACKER_IP_DESK']
    columns = [
        "Date", "Time", "Current", "Voltage", "Power", "Total", "ErrorCode"
    ]

    def __init__(self, ip_device=None, frequency_mesure=1):
        if ip_device is None:
            if EnergyTracker.IP_DEVICE is None:
                raise ValueError('Please put a valid IP adress for your TP-Link energy tracker device.')
            ip_device = EnergyTracker.IP_DEVICE
        self.ip_device = ip_device
        self.frequency_mesure = frequency_mesure

    def save(self, filename, line_csv):
        filename = filename + ".csv"

        if not os.path.isfile(filename):
            with open(filename, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(EnergyTracker.columns)

        with open(filename, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(line_csv)

    async def run_real_time(self):
        smart_plug = SmartPlug(self.ip_device)
        while 1:
            await smart_plug.update()

            smart_plug_real_time_data = smart_plug.emeter_realtime
            print(smart_plug_real_time_data)
            now = datetime.now()
            current_date, current_time = re.split(
                " ", now.strftime("%d-%m-%Y %H:%M:%S"))
            new_csv_line = [
                current_date, current_time, smart_plug_real_time_data['current'],
                smart_plug_real_time_data['voltage'], smart_plug_real_time_data['power'],
                smart_plug_real_time_data['total'], smart_plug_real_time_data['err_code']
            ]
            self.save(current_date, new_csv_line)
            time.sleep(self.frequency_mesure)

    async def get_daily_mesures(self):
        smart_plug = SmartPlug(self.ip_device)
        await smart_plug.update()
        smart_plug_daily_data = await smart_plug.get_emeter_daily()
        print(smart_plug_daily_data)
        return smart_plug_daily_data

    async def get_monthly_mesures(self):
        smart_plug = SmartPlug(self.ip_device)
        await smart_plug.update()
        smart_plug_monthly_data = await smart_plug.get_emeter_monthly()
        print(smart_plug_monthly_data)
        return smart_plug_monthly_data


if __name__ == "__main__":
    energy_tracker = EnergyTracker()
    asyncio.run(energy_tracker.run_real_time())
