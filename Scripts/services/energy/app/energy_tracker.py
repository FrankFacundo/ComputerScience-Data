"""
author: Frank Facundo
"""

import asyncio
import csv
import os
import re
import sys
import time

from datetime import datetime

from kasa import SmartPlug

from tools import execute_command


class EnergyTracker:
    """
    A class that tracks energy usage and saves data to a CSV file.
    """
    columns = [
        "Date", "Time", "Current", "Voltage", "Power", "Total", "ErrorCode"
    ]

    def __init__(self, ip_device=None, frequency_mesure=1):
        if ip_device is None:
            ip_devices = self.get_tp_link_ips()
            if not ip_devices:
                raise ValueError(
                    'Please put a valid IP address for your TP-Link energy tracker device.'
                )
            ip_device = ip_devices[0]
        self.ip_device = ip_device
        print(f'IP Device: {self.ip_device}')
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
                current_date, current_time,
                smart_plug_real_time_data['current'],
                smart_plug_real_time_data['voltage'],
                smart_plug_real_time_data['power'],
                smart_plug_real_time_data['total'],
                smart_plug_real_time_data['err_code']
            ]
            filename = f"{self.ip_device}_{current_date}"
            self.save(filename, new_csv_line)
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

    def get_tp_link_ips(self):
        command = "nmap -sP 192.168.1.0/24"
        nmap_info = execute_command(command, use_sudo=True)

        tp_link_pattern = re.compile(
            r'Nmap scan report for (\d+\.\d+\.\d+\.\d+).*?MAC Address: ((?:[0-9A-F]{2}:){5}[0-9A-F]{2}) \((.*?)\)',
            re.DOTALL)
        matches = tp_link_pattern.findall(nmap_info)
        tp_link_ips = [
            ip for ip, mac, company in matches
            if 'Tp-link Technologies' in company
        ]
        return tp_link_ips


if __name__ == "__main__":
    ip_device_tracker = None
    if len(sys.argv) > 1:
        ip_device_tracker = sys.argv[1]
    energy_tracker = EnergyTracker(ip_device=ip_device_tracker)
    asyncio.run(energy_tracker.run_real_time())
