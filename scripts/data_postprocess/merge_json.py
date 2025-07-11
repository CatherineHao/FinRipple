import os
import json


sp500_tickers = {
    '3M Company': 'MMM',
    'Abbott Laboratories': 'ABT',
    'AbbVie Inc.': 'ABBV',
    'ABIOMED Inc.': 'ABMD',
    'Accenture plc': 'ACN',
    'Activision Blizzard Inc.': 'ATVI',
    'Adobe Inc.': 'ADBE',
    'Advanced Micro Devices Inc.': 'AMD',
    'AES Corporation': 'AES',
    'AFLAC Incorporated': 'AFL',
    'Agilent Technologies Inc.': 'A',
    'Air Products and Chemicals Inc.': 'APD',
    'Akamai Technologies Inc.': 'AKAM',
    'Albemarle Corporation': 'ALB',
    'Alexandria Real Estate Equities Inc.': 'ARE',
    'Align Technology Inc.': 'ALGN',
    'Allegion plc': 'ALLE',
    'Alliant Energy Corp': 'LNT',
    'Allstate Corporation': 'ALL',
    'Alphabet Inc. Class A': 'GOOGL',
    'Alphabet Inc. Class C': 'GOOG',
    'Amazon.com Inc.': 'AMZN',
    'Ameren Corporation': 'AEE',
    'American Electric Power Company Inc.': 'AEP',
    'American Express Company': 'AXP',
    'American International Group Inc.': 'AIG',
    'American Tower Corporation': 'AMT',
    'American Water Works Company Inc.': 'AWK',
    'Ameriprise Financial Inc.': 'AMP',
    'Amgen Inc.': 'AMGN',
    'Analog Devices Inc.': 'ADI',
    'Anthem Inc.': 'ANTM',
    'Aon plc': 'AON',
    'Apple Inc.': 'AAPL',
    'Applied Materials Inc.': 'AMAT',
    'Aptiv PLC': 'APTV',
    'Archer-Daniels-Midland Company': 'ADM',
    'Arthur J. Gallagher & Co.': 'AJG',
    'AT&T Inc.': 'T',
    'Autodesk Inc.': 'ADSK',
    'Automatic Data Processing Inc.': 'ADP',
    'AvalonBay Communities Inc.': 'AVB',
    'Baker Hughes Company': 'BKR',
    'Ball Corporation': 'BALL',
    'Bank of America Corporation': 'BAC',
    'Bank of New York Mellon Corporation': 'BK',
    'Bath & Body Works Inc.': 'BBWI',
    'Baxter International Inc.': 'BAX',
    'Becton, Dickinson and Company': 'BDX',
    'Berkshire Hathaway Inc. Class B': 'BRK.B',
    'Best Buy Co. Inc.': 'BBY',
    'Biogen Inc.': 'BIIB',
    'BlackRock Inc.': 'BLK',
    'Boeing Company': 'BA',
    'Booking Holdings Inc.': 'BKNG',
    'BorgWarner Inc.': 'BWA',
    'Boston Properties Inc.': 'BXP',
    'Boston Scientific Corporation': 'BSX',
    'Bristol-Myers Squibb Company': 'BMY',
    'Broadcom Inc.': 'AVGO',
    'Broadridge Financial Solutions Inc.': 'BR',
    'Brown & Brown Inc.': 'BRO',
    'Brown-Forman Corporation Class B': 'BF.B',
    'C.H. Robinson Worldwide Inc.': 'CHRW',
    'Cadence Design Systems Inc.': 'CDNS',
    'Caesars Entertainment Inc.': 'CZR',
    'Campbell Soup Company': 'CPB',
    'Capital One Financial Corporation': 'COF',
    'Cardinal Health Inc.': 'CAH',
    'CarMax Inc.': 'KMX',
    'Carnival Corporation': 'CCL',
    'Carrier Global Corporation': 'CARR',
    'Catalent Inc.': 'CTLT',
    'Caterpillar Inc.': 'CAT',
    'Cboe Global Markets Inc.': 'CBOE',
    'CBRE Group Inc.': 'CBRE',
    'CDW Corporation': 'CDW',
    'Celanese Corporation': 'CE',
    'Centene Corporation': 'CNC',
    'CenterPoint Energy Inc.': 'CNP',
    'Ceridian HCM Holding Inc.': 'CDAY',
    'CF Industries Holdings Inc.': 'CF',
    'Charles Schwab Corporation': 'SCHW',
    'Charter Communications Inc.': 'CHTR',
    'Chevron Corporation': 'CVX',
    'Chipotle Mexican Grill Inc.': 'CMG',
    'Chubb Limited': 'CB',
    'Church & Dwight Co. Inc.': 'CHD',
    'Cigna Group': 'CI',
    'Cincinnati Financial Corporation': 'CINF',
    'Cintas Corporation': 'CTAS',
    'Cisco Systems Inc.': 'CSCO',
    'Citigroup Inc.': 'C',
    'Citizens Financial Group Inc.': 'CFG',
    'Clorox Company': 'CLX',
    'CME Group Inc.': 'CME',
    'CMS Energy Corporation': 'CMS',
    'Coca-Cola Company': 'KO',
    'Cognizant Technology Solutions Corporation': 'CTSH',
    'Colgate-Palmolive Company': 'CL',
    'Comcast Corporation': 'CMCSA',
    'Comerica Incorporated': 'CMA',
    'Conagra Brands Inc.': 'CAG',
    'ConocoPhillips': 'COP',
    'Consolidated Edison Inc.': 'ED',
    'Constellation Brands Inc.': 'STZ',
    'Constellation Energy Corporation': 'CEG',
    'Copart Inc.': 'CPRT',
    'Corning Inc.': 'GLW',
    'Corteva Inc.': 'CTVA',
    'Costco Wholesale Corporation': 'COST',
    'Coterra Energy Inc.': 'CTRA',
    'Crown Castle Inc.': 'CCI',
    'CSX Corporation': 'CSX',
    'Cummins Inc.': 'CMI',
    'CVS Health Corporation': 'CVS',
    'D.R. Horton Inc.': 'DHI',
    'Danaher Corporation': 'DHR',
    'Darden Restaurants Inc.': 'DRI',
    'DaVita Inc.': 'DVA',
    'Deere & Company': 'DE',
    'Delta Air Lines Inc.': 'DAL',
    'DENTSPLY SIRONA Inc.': 'XRAY',
    'Devon Energy Corporation': 'DVN',
    'DexCom Inc.': 'DXCM',
    'Diamondback Energy Inc.': 'FANG',
    'Digital Realty Trust Inc.': 'DLR',
    'Discover Financial Services': 'DFS',
    'Dollar General Corporation': 'DG',
    'Dollar Tree Inc.': 'DLTR',
    'Dominion Energy Inc.': 'D',
    'Domino\'s Pizza Inc.': 'DPZ',
    'Dover Corporation': 'DOV',
    'Dow Inc.': 'DOW',
    'DTE Energy Company': 'DTE',
    'Duke Energy Corporation': 'DUK',
    'DuPont de Nemours Inc.': 'DD',
    'DXC Technology Company': 'DXC',
    'Eastman Chemical Company': 'EMN',
    'Eaton Corporation plc': 'ETN',
    'eBay Inc.': 'EBAY',
    'Ecolab Inc.': 'ECL',
    'Edison International': 'EIX',
    'Edwards Lifesciences Corporation': 'EW',
    'Electronic Arts Inc.': 'EA',
    'Emerson Electric Co.': 'EMR',
    'Entergy Corporation': 'ETR',
    'Enphase Energy Inc.': 'ENPH',
    'EOG Resources Inc.': 'EOG',
    'EPAM Systems Inc.': 'EPAM',
    'Equifax Inc.': 'EFX',
    'Equinix Inc.': 'EQIX',
    'Equity Residential': 'EQR',
    'Essex Property Trust Inc.': 'ESS',
    'Estee Lauder Companies Inc.': 'EL',
    'Etsy Inc.': 'ETSY',
    'Evergy Inc.': 'EVRG',
    'Eversource Energy': 'ES',
    'Exelon Corporation': 'EXC',
    'Expedia Group Inc.': 'EXPE',
    'Expeditors International of Washington Inc.': 'EXPD',
    'Extra Space Storage Inc.': 'EXR',
    'Exxon Mobil Corporation': 'XOM',
    'F5 Inc.': 'FFIV',
    'Fastenal Company': 'FAST',
    'Federal Realty Investment Trust': 'FRT',
    'FedEx Corporation': 'FDX',
    'Fidelity National Information Services Inc.': 'FIS',
    'Fifth Third Bancorp': 'FITB',
    'FirstEnergy Corp.': 'FE',
    'Fiserv Inc.': 'FISV',
    'FleetCor Technologies Inc.': 'FLT',
    'Ford Motor Company': 'F',
    'Fortinet Inc.': 'FTNT',
    'Fortive Corporation': 'FTV',
    'Fortune Brands Innovations Inc.': 'FBIN',
    'Fox Corporation Class A': 'FOXA',
    'Fox Corporation Class B': 'FOX',
    'Franklin Resources Inc.': 'BEN',
    'Freeport-McMoRan Inc.': 'FCX',
    'Garmin Ltd.': 'GRMN',
    'Gartner Inc.': 'IT',
    'General Dynamics Corporation': 'GD',
    'General Electric Company': 'GE',
    'General Mills Inc.': 'GIS',
    'General Motors Company': 'GM',
    'Genuine Parts Company': 'GPC',
    'Gilead Sciences Inc.': 'GILD',
    'Global Payments Inc.': 'GPN',
    'Goldman Sachs Group Inc.': 'GS',
    'Halliburton Company': 'HAL',
    'Hartford Financial Services Group Inc.': 'HIG',
    'Hasbro Inc.': 'HAS',
    'HCA Healthcare Inc.': 'HCA',
    'Healthpeak Properties Inc.': 'PEAK',
    'Henry Schein Inc.': 'HSIC',
    'Hess Corporation': 'HES',
    'Hewlett Packard Enterprise Company': 'HPE',
    'Hilton Worldwide Holdings Inc.': 'HLT',
    'Hologic Inc.': 'HOLX',
    'Home Depot Inc.': 'HD',
    'Honeywell International Inc.': 'HON',
    'Hormel Foods Corporation': 'HRL',
    'Host Hotels & Resorts Inc.': 'HST',
    'Howmet Aerospace Inc.': 'HWM',
    'HP Inc.': 'HPQ',
    'Humana Inc.': 'HUM',
    'Huntington Bancshares Incorporated': 'HBAN',
    'Huntington Ingalls Industries Inc.': 'HII',
    'IDEX Corporation': 'IEX',
    'IDEXX Laboratories Inc.': 'IDXX',
    'Illinois Tool Works Inc.': 'ITW',
    'Illumina Inc.': 'ILMN',
    'Incyte Corporation': 'INCY',
    'Ingersoll Rand Inc.': 'IR',
    'Intel Corporation': 'INTC',
    'Intercontinental Exchange Inc.': 'ICE',
    'International Business Machines Corporation': 'IBM',
    'International Flavors & Fragrances Inc.': 'IFF',
    'International Paper Company': 'IP',
    'Interpublic Group of Companies Inc.': 'IPG',
    'Intuit Inc.': 'INTU',
    'Intuitive Surgical Inc.': 'ISRG',
    'Iron Mountain Incorporated': 'IRM',
    'IQVIA Holdings Inc.': 'IQV',
    'J.B. Hunt Transport Services Inc.': 'JBHT',
    'Jack Henry & Associates Inc.': 'JKHY',
    'Jacobs Solutions Inc.': 'J',
    'Johnson & Johnson': 'JNJ',
    'Johnson Controls International plc': 'JCI',
    'JPMorgan Chase & Co.': 'JPM',
    'Juniper Networks Inc.': 'JNPR',
    'Keurig Dr Pepper Inc.': 'KDP',
    'KeyCorp': 'KEY',
    'Keysight Technologies Inc.': 'KEYS',
    'Kimberly-Clark Corporation': 'KMB',
    'Kinder Morgan Inc.': 'KMI',
    'KLA Corporation': 'KLAC',
    'Kraft Heinz Company': 'KHC',
    'Kroger Co.': 'KR',
    'L3Harris Technologies Inc.': 'LHX',
    'Lam Research Corporation': 'LRCX',
    'Las Vegas Sands Corp.': 'LVS',
    'Leidos Holdings Inc.': 'LDOS',
    'Lennar Corporation': 'LEN',
    'Lilly (Eli) & Co.': 'LLY',
    'Lincoln National Corporation': 'LNC',
    'Linde plc': 'LIN',
    'Live Nation Entertainment Inc.': 'LYV',
    'LKQ Corporation': 'LKQ',
    'Lockheed Martin Corporation': 'LMT',
    'Loews Corporation': 'L',
    'Lowe\'s Companies Inc.': 'LOW',
    'Lumen Technologies Inc.': 'LUMN',
    'LyondellBasell Industries N.V.': 'LYB',
    'M&T Bank Corporation': 'MTB',
    'Marathon Oil Corporation': 'MRO',
    'Marathon Petroleum Corporation': 'MPC',
    'MarketAxess Holdings Inc.': 'MKTX',
    'Marriott International Inc.': 'MAR',
    'Marsh & McLennan Companies Inc.': 'MMC',
    'Martin Marietta Materials Inc.': 'MLM',
    'Masco Corporation': 'MAS',
    'Mastercard Incorporated': 'MA',
    'Match Group Inc.': 'MTCH',
    'McCormick & Company Incorporated': 'MKC',
    'McDonald\'s Corporation': 'MCD',
    'McKesson Corporation': 'MCK',
    'Medtronic plc': 'MDT',
    'Merck & Co. Inc.': 'MRK',
    'Meta Platforms Inc.': 'META',
    'MetLife Inc.': 'MET',
    'Mettler-Toledo International Inc.': 'MTD',
    'Micron Technology Inc.': 'MU',
    'Microsoft Corporation': 'MSFT',
    'Mid-America Apartment Communities Inc.': 'MAA',
    'Moderna Inc.': 'MRNA',
    'Mohawk Industries Inc.': 'MHK',
    'Molson Coors Beverage Company': 'TAP',
    'Monolithic Power Systems Inc.': 'MPWR',
    'Mondelez International Inc.': 'MDLZ',
    'Monster Beverage Corporation': 'MNST',
    'Moody\'s Corporation': 'MCO',
    'Morgan Stanley': 'MS',
    'Motorola Solutions Inc.': 'MSI',
    'Nasdaq Inc.': 'NDAQ',
    'Netflix Inc.': 'NFLX',
    'Newell Brands Inc.': 'NWL',
    'Newmont Corporation': 'NEM',
    'News Corporation Class A': 'NWSA',
    'News Corporation Class B': 'NWS',
    'NextEra Energy Inc.': 'NEE',
    'Nike Inc.': 'NKE',
    'NiSource Inc.': 'NI',
    'Norfolk Southern Corporation': 'NSC',
    'Northern Trust Corporation': 'NTRS',
    'Northrop Grumman Corporation': 'NOC',
    'Nucor Corporation': 'NUE',
    'NVIDIA Corporation': 'NVDA',
    'NVR Inc.': 'NVR',
    'Occidental Petroleum Corporation': 'OXY',
    'Omnicom Group Inc.': 'OMC',
    'Oracle Corporation': 'ORCL',
    'O\'Reilly Automotive Inc.': 'ORLY',
    'Otis Worldwide Corporation': 'OTIS',
    'PACCAR Inc.': 'PCAR',
    'Packaging Corporation of America': 'PKG',
    'Paramount Global Class B': 'PARA',
    'Paychex Inc.': 'PAYX',
    'Paycom Software Inc.': 'PAYC',
    'PayPal Holdings Inc.': 'PYPL',
    'Pentair plc': 'PNR',
    'PepsiCo Inc.': 'PEP',
    'PerkinElmer Inc.': 'PKI',
    'Pfizer Inc.': 'PFE',
    'Philip Morris International Inc.': 'PM',
    'Phillips 66': 'PSX',
    'Pioneer Natural Resources Company': 'PXD',
    'PNC Financial Services Group Inc.': 'PNC',
    'PPG Industries Inc.': 'PPG',
    'PPL Corporation': 'PPL',
    'Principal Financial Group Inc.': 'PFG',
    'Procter & Gamble Company': 'PG',
    'Progressive Corporation': 'PGR',
    'Prologis Inc.': 'PLD',
    'Prudential Financial Inc.': 'PRU',
    'Public Service Enterprise Group Incorporated': 'PEG',
    'Public Storage': 'PSA',
    'PulteGroup Inc.': 'PHM',
    'Qorvo Inc.': 'QRVO',
    'Quanta Services Inc.': 'PWR',
    'QUALCOMM Incorporated': 'QCOM',
    'Quest Diagnostics Incorporated': 'DGX',
    'Ralph Lauren Corporation': 'RL',
    'Raymond James Financial Inc.': 'RJF',
    'Raytheon Technologies Corporation': 'RTX',
    'Realty Income Corporation': 'O',
    'Regency Centers Corporation': 'REG',
    'Regeneron Pharmaceuticals Inc.': 'REGN',
    'Regions Financial Corporation': 'RF',
    'Republic Services Inc.': 'RSG',
    'ResMed Inc.': 'RMD',
    'Robert Half International Inc.': 'RHI',
    'Rockwell Automation Inc.': 'ROK',
    'Rollins Inc.': 'ROL',
    'Roper Technologies Inc.': 'ROP',
    'Ross Stores Inc.': 'ROST',
    'S&P Global Inc.': 'SPGI',
    'Salesforce Inc.': 'CRM',
    'Schlumberger Limited': 'SLB',
    'Seagate Technology Holdings plc': 'STX',
    'Sealed Air Corporation': 'SEE',
    'Sempra Energy': 'SRE',
    'ServiceNow Inc.': 'NOW',
    'Sherwin-Williams Company': 'SHW',
    'Simon Property Group Inc.': 'SPG',
    'Skyworks Solutions Inc.': 'SWKS',
    'Snap-on Incorporated': 'SNA',
    'Southern Company': 'SO',
    'Southwest Airlines Co.': 'LUV',
    'Stanley Black & Decker Inc.': 'SWK',
    'Starbucks Corporation': 'SBUX',
    'State Street Corporation': 'STT',
    'Steel Dynamics Inc.': 'STLD',
    'Stryker Corporation': 'SYK',
    'Synchrony Financial': 'SYF',
    'Synopsys Inc.': 'SNPS',
    'Sysco Corporation': 'SYY',
    'T-Mobile US Inc.': 'TMUS',
    'Target Corporation': 'TGT',
    'TE Connectivity Ltd.': 'TEL',
    'Teledyne Technologies Incorporated': 'TDY',
    'Teleflex Incorporated': 'TFX',
    'Teradyne Inc.': 'TER',
    'Tesla Inc.': 'TSLA',
    'Texas Instruments Incorporated': 'TXN',
    'Textron Inc.': 'TXT',
    'Thermo Fisher Scientific Inc.': 'TMO',
    'TJX Companies Inc.': 'TJX',
    'Tractor Supply Company': 'TSCO',
    'Trane Technologies plc': 'TT',
    'TransDigm Group Incorporated': 'TDG',
    'Travelers Companies Inc.': 'TRV',
    'Truist Financial Corporation': 'TFC',
    'Tyler Technologies Inc.': 'TYL',
    'Tyson Foods Inc.': 'TSN',
    'UDR Inc.': 'UDR',
    'Ulta Beauty Inc.': 'ULTA',
    'Union Pacific Corporation': 'UNP',
    'United Airlines Holdings Inc.': 'UAL',
    'United Parcel Service Inc.': 'UPS',
    'United Rentals Inc.': 'URI',
    'UnitedHealth Group Incorporated': 'UNH',
    'Universal Health Services Inc. Class B': 'UHS',
    'Valero Energy Corporation': 'VLO',
    'Ventas Inc.': 'VTR',
    'Verisk Analytics Inc.': 'VRSK',
    'Verizon Communications Inc.': 'VZ',
    'Vertex Pharmaceuticals Incorporated': 'VRTX',
    'VF Corporation': 'VFC',
    'ViacomCBS Inc.': 'VIAC',
    'Visa Inc.': 'V',
    'Vulcan Materials Company': 'VMC',
    'W.W. Grainger Inc.': 'GWW',
    'Walgreens Boots Alliance Inc.': 'WBA',
    'Walmart Inc.': 'WMT',
    'Waste Management Inc.': 'WM',
    'Waters Corporation': 'WAT',
    'Wells Fargo & Company': 'WFC',
    'Welltower Inc.': 'WELL',
    'West Pharmaceutical Services Inc.': 'WST',
    'Western Digital Corporation': 'WDC',
    'Western Union Company': 'WU',
    'WestRock Company': 'WRK',
    'Weyerhaeuser Company': 'WY',
    'Whirlpool Corporation': 'WHR',
    'Williams Companies Inc.': 'WMB',
    'Willis Towers Watson Public Limited Company': 'WTW',
    'Wynn Resorts Limited': 'WYNN',
    'Xcel Energy Inc.': 'XEL',
    'Xylem Inc.': 'XYL',
    'Yum! Brands Inc.': 'YUM',
    'Zimmer Biomet Holdings Inc.': 'ZBH',
    'Zions Bancorporation N.A.': 'ZION',
    'Zoetis Inc.': 'ZTS',
}

import os
import json
import pandas as pd
from collections import defaultdict


class SP500DataProcessor:
    """Processes JSON files containing company impact data and saves to CSV."""

    def __init__(self, folder_path: str, output_file: str):
        """
        Initializes the data processor with the folder path and output file.

        Args:
            folder_path (str): Path to the folder containing JSON files.
            output_file (str): Name of the output JSON file to store merged data.
        """
        self.folder_path = folder_path
        self.output_file = output_file
        self.sp500_tickers = sp500_tickers
        self.all_data = []

    def process_json_files(self):
        """Processes all JSON files in the specified folder."""
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(self.folder_path, filename)
                print(f"Processing file: {file_path}")
                self._process_single_file(file_path)

        # Save merged data to a JSON file
        with open(self.output_file, 'w', encoding='utf-8') as json_file:
            json.dump(self.all_data, json_file, indent=4)

        print(f"Processing completed. Merged data saved to '{self.output_file}'.")

    def _process_single_file(self, file_path: str):
        """
        Processes a single JSON file and updates the overall data.

        Args:
            file_path (str): Path to the JSON file.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON file {file_path}: {e}")
            return

        for item in data:
            self._process_item(item)

    def _process_item(self, item: dict):
        """
        Processes a single item from a JSON file.

        Args:
            item (dict): Dictionary representing a JSON item.
        """
        try:
            response_data = json.loads(item['response'])
            impact_analysis = response_data.get('impact_analysis', {})
            affected_companies = impact_analysis.get('affected_companies', [])

            new_affected_companies = []
            for company in affected_companies:
                company_name = company.get('name')
                ticker = self.sp500_tickers.get(company_name)
                if ticker:
                    company['name'] = ticker
                    new_affected_companies.append(company)
                else:
                    print(f"Company '{company_name}' not in S&P 500, removing from the list.")

            if new_affected_companies:
                impact_analysis['affected_companies'] = new_affected_companies
                response_data['impact_analysis'] = impact_analysis
                item['response'] = json.dumps(response_data, indent=4)
                self.all_data.append(item)
            else:
                print(f"No affected companies left in item with prompt '{item.get('prompt')}', removing the item.")

        except json.JSONDecodeError as e:
            print(f"JSONDecodeError in item with prompt '{item.get('prompt')}': {e}")

    def calculate_date_company_impact(self) -> pd.DataFrame:
        """
        Calculates the impact scores for each company on each date.

        Returns:
            pd.DataFrame: A DataFrame containing the impact scores for each company by date.
        """
        date_company_impact = defaultdict(lambda: defaultdict(float))

        for item in self.all_data:
            date = item.get('date')
            if not date:
                print(f"Item with prompt '{item.get('prompt')}' has no date, skipping.")
                continue

            try:
                response_data = json.loads(item['response'])
                impact_analysis = response_data.get('impact_analysis', {})
                affected_companies = impact_analysis.get('affected_companies', [])

                for company in affected_companies:
                    ticker = company.get('name')  # This should be the ticker name
                    impact_score = company.get('impact_score', 0)
                    if ticker:
                        date_company_impact[date][ticker] += impact_score

            except json.JSONDecodeError as e:
                print(f"JSONDecodeError in item with prompt '{item.get('prompt')}': {e}")

        # Convert nested dictionary to DataFrame
        records = []
        for date, impacts in date_company_impact.items():
            record = {'date': date}
            record.update(impacts)
            records.append(record)
        df = pd.DataFrame(records)

        # Fill missing values with 0
        df = df.fillna(0)

        # Adjust columns order to have 'date' first
        columns = ['date'] + sorted([col for col in df.columns if col != 'date'])
        df = df[columns]

        return df

    def save_to_csv(self, df: pd.DataFrame, csv_filename: str):
        """
        Saves the DataFrame to a CSV file.

        Args:
            df (pd.DataFrame): DataFrame containing the impact data.
            csv_filename (str): Name of the output CSV file.
        """
        df.to_csv(csv_filename, index=False)
        print(f"CSV file '{csv_filename}' has been generated.")


# Usage Example:
if __name__ == "__main__":
    folder_path = 'jsons'  # Replace with your folder path
    output_json_file = 'merged_output.json'
    output_csv_file = 'output.csv'

    processor = SP500DataProcessor(folder_path, output_json_file)
    processor.process_json_files()

    # Calculate impact and save to CSV
    df = processor.calculate_date_company_impact()
    processor.save_to_csv(df, output_csv_file)
