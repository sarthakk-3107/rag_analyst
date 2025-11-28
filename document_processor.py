"""
Document Processor for SEC Filings
Handles parsing, cleaning, and structuring of 10-K documents

FE524 Project - Phase 1
"""

import re
import os
from pathlib import Path
from typing import List, Dict, Tuple
from bs4 import BeautifulSoup
from dataclasses import dataclass
import json

@dataclass
class SECSection:
    """Represents a section from an SEC filing"""
    section_number: str
    section_title: str
    content: str
    company: str
    filing_date: str
    form_type: str

class SECDocumentProcessor:
    """Process and parse SEC 10-K filings"""

    # Standard 10-K sections
    SECTIONS = {
        'Item 1': 'Business',
        'Item 1A': 'Risk Factors',
        'Item 1B': 'Unresolved Staff Comments',
        'Item 2': 'Properties',
        'Item 3': 'Legal Proceedings',
        'Item 4': 'Mine Safety Disclosures',
        'Item 5': 'Market for Registrant\'s Common Equity',
        'Item 6': 'Selected Financial Data',
        'Item 7': 'Management\'s Discussion and Analysis',
        'Item 7A': 'Quantitative and Qualitative Disclosures About Market Risk',
        'Item 8': 'Financial Statements and Supplementary Data',
        'Item 9': 'Changes in and Disagreements with Accountants',
        'Item 9A': 'Controls and Procedures',
        'Item 9B': 'Other Information',
        'Item 10': 'Directors, Executive Officers and Corporate Governance',
        'Item 11': 'Executive Compensation',
        'Item 12': 'Security Ownership',
        'Item 13': 'Certain Relationships and Related Transactions',
        'Item 14': 'Principal Accounting Fees and Services',
        'Item 15': 'Exhibits, Financial Statement Schedules'
    }

    def __init__(self):
        self.sections_cache = {}

    def read_filing(self, filepath: str) -> str:
        """Read SEC filing from file"""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    def clean_html(self, html_content: str) -> str:
        """Clean HTML and extract text"""
        soup = BeautifulSoup(html_content, 'lxml')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text
        text = soup.get_text()

        # Clean whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        return text

    def extract_metadata(self, content: str) -> Dict[str, str]:
        """Extract metadata from filing"""
        metadata = {
            'company_name': '',
            'cik': '',
            'filing_date': '',
            'fiscal_year_end': ''
        }

        # Extract company name
        company_match = re.search(r'COMPANY CONFORMED NAME:\s+(.+)', content)
        if company_match:
            metadata['company_name'] = company_match.group(1).strip()

        # Extract CIK
        cik_match = re.search(r'CENTRAL INDEX KEY:\s+(\d+)', content)
        if cik_match:
            metadata['cik'] = cik_match.group(1).strip()

        # Extract filing date
        date_match = re.search(r'FILED AS OF DATE:\s+(\d+)', content)
        if date_match:
            date_str = date_match.group(1)
            metadata['filing_date'] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

        return metadata

    def split_into_sections(self, text: str) -> Dict[str, str]:
        """Split 10-K into standard sections"""
        sections = {}

        # Create patterns for each section
        for item_num, item_title in self.SECTIONS.items():
            # Multiple pattern variations to catch different formats
            patterns = [
                rf"(?:^|\n)\s*{re.escape(item_num)}[\.\-\s]+{re.escape(item_title)}",
                rf"(?:^|\n)\s*{re.escape(item_num.upper())}[\.\-\s]+{re.escape(item_title.upper())}",
                rf"(?:^|\n)\s*{re.escape(item_num)}[\.\-\s:]+.*?{re.escape(item_title)}"
            ]

            matches = []
            for pattern in patterns:
                matches.extend(list(re.finditer(pattern, text, re.IGNORECASE)))

            if matches:
                # Use the first match found
                start_pos = matches[0].end()

                # Find the start of next section
                next_section_found = False
                remaining_items = [k for k in self.SECTIONS.keys() if k > item_num]

                for next_item in remaining_items:
                    next_patterns = [
                        rf"(?:^|\n)\s*{re.escape(next_item)}[\.\-\s]+",
                        rf"(?:^|\n)\s*{re.escape(next_item.upper())}[\.\-\s]+"
                    ]

                    for next_pattern in next_patterns:
                        next_match = re.search(next_pattern, text[start_pos:], re.IGNORECASE)
                        if next_match:
                            end_pos = start_pos + next_match.start()
                            sections[item_num] = text[start_pos:end_pos].strip()
                            next_section_found = True
                            break

                    if next_section_found:
                        break

                # If no next section found, take rest of document (but limit to 100k chars)
                if not next_section_found:
                    sections[item_num] = text[start_pos:start_pos + 100000].strip()

        return sections

    def extract_financial_tables(self, text: str) -> List[Dict[str, str]]:
        """Extract financial tables and data"""
        tables = []

        # Common financial statement headers
        headers = [
            'CONSOLIDATED STATEMENTS OF INCOME',
            'CONSOLIDATED STATEMENTS OF OPERATIONS',
            'CONSOLIDATED BALANCE SHEETS',
            'CONSOLIDATED STATEMENTS OF CASH FLOWS',
            'STATEMENTS OF COMPREHENSIVE INCOME'
        ]

        for header in headers:
            pattern = rf"{header}.*?(?=CONSOLIDATED|\Z)"
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)

            for match in matches:
                table_text = match.group(0)
                if len(table_text) > 100:  # Minimum table size
                    tables.append({
                        'type': header,
                        'content': table_text[:5000]  # Limit size
                    })

        return tables

    def extract_financial_metrics(self, text: str) -> Dict[str, str]:
        """Extract key financial metrics using regex patterns"""
        metrics = {}

        # Revenue patterns
        revenue_patterns = [
            r'(?:Total\s+)?(?:Net\s+)?(?:Revenue|Sales).*?\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion)?',
            r'Revenue.*?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)'
        ]

        for pattern in revenue_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metrics['revenue'] = match.group(1)
                break

        # Net income patterns
        income_patterns = [
            r'Net\s+Income.*?\$?\s*([\d,]+(?:\.\d+)?)',
            r'Net\s+(?:Earnings|Income).*?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)'
        ]

        for pattern in income_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metrics['net_income'] = match.group(1)
                break

        # Operating income
        op_income_match = re.search(
            r'Operating\s+Income.*?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
            text,
            re.IGNORECASE
        )
        if op_income_match:
            metrics['operating_income'] = op_income_match.group(1)

        return metrics

    def process_filing(self, filepath: str) -> Tuple[List[SECSection], Dict]:
        """Main processing pipeline for a 10-K filing"""

        # Read raw content
        raw_content = self.read_filing(filepath)

        # Extract metadata
        metadata = self.extract_metadata(raw_content)

        # Clean HTML
        clean_text = self.clean_html(raw_content)

        # Split into sections
        sections_dict = self.split_into_sections(clean_text)

        # Extract financial metrics
        financial_metrics = self.extract_financial_metrics(clean_text)
        metadata['financial_metrics'] = financial_metrics

        # Create SECSection objects
        sections = []
        for item_num, content in sections_dict.items():
            section = SECSection(
                section_number=item_num,
                section_title=self.SECTIONS[item_num],
                content=content,
                company=metadata.get('company_name', 'Unknown'),
                filing_date=metadata.get('filing_date', ''),
                form_type='10-K'
            )
            sections.append(section)

        return sections, metadata

    def chunk_section(self, section: SECSection, chunk_size: int = 1000,
                     overlap: int = 200) -> List[Dict]:
        """Chunk a section into smaller pieces for RAG"""
        words = section.content.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if len(chunk_words) < 50:  # Skip very small chunks
                continue

            chunk_text = ' '.join(chunk_words)
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    'section': section.section_number,
                    'section_title': section.section_title,
                    'company': section.company,
                    'filing_date': section.filing_date,
                    'form_type': section.form_type,
                    'chunk_index': len(chunks)
                }
            })

        return chunks

    def save_processed_filing(self, sections: List[SECSection],
                             metadata: Dict, output_dir: str):
        """Save processed filing to disk"""
        os.makedirs(output_dir, exist_ok=True)

        # Save metadata
        with open(f"{output_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save each section
        for section in sections:
            filename = f"{section.section_number.replace(' ', '_')}.txt"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Section: {section.section_number} - {section.section_title}\n")
                f.write(f"Company: {section.company}\n")
                f.write(f"Filing Date: {section.filing_date}\n")
                f.write("=" * 80 + "\n\n")
                f.write(section.content)

# Example usage and testing
if __name__ == "__main__":
    processor = SECDocumentProcessor()

    # Process a 10-K filing
    filing_path = "sec-edgar-filings/AAPL/10-K/0000320193-23-000077.txt"

    if os.path.exists(filing_path):
        print("Processing 10-K filing...")
        sections, metadata = processor.process_filing(filing_path)

        print(f"\nCompany: {metadata.get('company_name')}")
        print(f"Filing Date: {metadata.get('filing_date')}")
        print(f"\nExtracted {len(sections)} sections:")

        for section in sections:
            print(f"  - {section.section_number}: {section.section_title}")
            print(f"    Length: {len(section.content)} characters")

        # Chunk all sections
        all_chunks = []
        for section in sections:
            chunks = processor.chunk_section(section)
            all_chunks.extend(chunks)

        print(f"\nTotal chunks created: {len(all_chunks)}")

        # Save processed data
        output_dir = f"data/processed/{metadata.get('company_name', 'company')}"
        processor.save_processed_filing(sections, metadata, output_dir)
        print(f"Saved to: {output_dir}")
    else:
        print(f"File not found: {filing_path}")
        print("Please download SEC filings first using the main app.")
