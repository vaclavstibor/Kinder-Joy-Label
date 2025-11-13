import streamlit as st
import pandas as pd
import os
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Kinder Joy Labeling App",
    page_icon="🎁",
    layout="wide"
)

# Toy list (24 toys)
TOYS = [
    "Nancy", "Mike", "Lucas", "Demogordon", "Steve", "Eleven", "Vecna", 
    "Eleven Down", "Max Down", "Eleven clip", "Demogordon pen", 
    "Steven and Robin pen", "Eica kabel", "Demogordon clip", "Will", "Max", 
    "Dustin", "Hopper", "Will Donw", "Steve Down", "Eddie Down", 
    "Dustin Down", "Hopper Down", "Robin Down"
]

# CSV file paths
CSV_FILE = "labeled_data.csv"
BACKUP_DIR = "backups"
BACKUP_FILE = os.path.join(BACKUP_DIR, "backup_data.csv")

def ensure_backup_dir():
    """Create backup directory if it doesn't exist"""
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)

@st.cache_data(ttl=5)  # Cache for 5 seconds - balances freshness with performance on Streamlit Cloud
def load_existing_data():
    """Load existing labeled data from CSV"""
    if os.path.exists(CSV_FILE):
        try:
            # Read CSV with error handling for concurrent access
            df = pd.read_csv(CSV_FILE)
            # Filter out empty rows (rows with all NaN or empty values)
            df = df.dropna(how='all')
            # Filter out rows where all values are empty strings
            df = df[~(df.astype(str).apply(lambda x: x.str.strip().eq('')).all(axis=1))]
            return df
        except Exception as e:
            # If file is locked or corrupted, return empty DataFrame
            return pd.DataFrame(columns=["timestamp", "toy", "balls_code", "toy_code", "location_state"])
    return pd.DataFrame(columns=["timestamp", "toy", "balls_code", "toy_code", "location_state"])

def save_backup(new_row):
    """Save individual submitted item to backup CSV"""
    ensure_backup_dir()
    
    # Create DataFrame from new row
    new_df = pd.DataFrame([new_row])
    
    # Check if backup file exists
    if os.path.exists(BACKUP_FILE):
        # Append to existing backup file
        try:
            backup_df = pd.read_csv(BACKUP_FILE)
            backup_df = pd.concat([backup_df, new_df], ignore_index=True)
            backup_df.to_csv(BACKUP_FILE, index=False)
        except Exception as e:
            # If reading fails, create new backup file
            new_df.to_csv(BACKUP_FILE, index=False)
    else:
        # Create new backup file
        new_df.to_csv(BACKUP_FILE, index=False)
    
    # Also create a timestamped individual backup file for extra safety
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    individual_backup = os.path.join(BACKUP_DIR, f"item_{timestamp}.csv")
    new_df.to_csv(individual_backup, index=False)

def save_data(df):
    """Save DataFrame to CSV and create full backup"""
    # Filter out empty rows before saving
    df = df.dropna(how='all')
    df = df[~(df.astype(str).apply(lambda x: x.str.strip().eq('')).all(axis=1))]
    
    # Remove duplicates based on all columns to prevent duplicate entries
    df = df.drop_duplicates(subset=["timestamp", "toy", "balls_code", "toy_code", "location_state"], keep='last')
    
    # Save main CSV file with error handling for concurrent writes
    try:
        df.to_csv(CSV_FILE, index=False)
    except Exception as e:
        # If write fails, try again after a short delay (handles concurrent access on Streamlit Cloud)
        time.sleep(0.1)
        df.to_csv(CSV_FILE, index=False)
    
    # Create timestamped full backup
    ensure_backup_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_backup = os.path.join(BACKUP_DIR, f"full_backup_{timestamp}.csv")
    df.to_csv(full_backup, index=False)

def main():
    # Centered title
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 2rem;'>Stranger Things Labeling Dataset</h1>",
        unsafe_allow_html=True
    )

    # Description
    st.markdown(
        "<p style='text-align: center; margin-bottom: 2rem;'>This app is used to make labeled Stranger Things Kinder Joy toys dataset which will be used to make statistical predictive model.</p>",
        unsafe_allow_html=True
    )

    # Refresh button to reload data
    col_refresh, _ = st.columns([1, 10])
    with col_refresh:
        if st.button("🔄 Refresh Data", help="Reload data to see latest submissions from all users"):
            load_existing_data.clear()  # Clear cache to force reload
            st.rerun()  # Rerun to refresh the page
    
    # Load existing data
    df = load_existing_data()
    
    # Main form
    with st.form("labeling_form", clear_on_submit=True):
        # All inputs in one row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # 1. Balls code
            balls_code = st.text_input(
                "Balls Code *",
                help="Must be at least 1 character",
                placeholder="Enter balls code"
            )
        
        with col2:
            # 2. Toy code
            toy_code = st.text_input(
                "Toy Code *",
                help="Must be at least 1 character",
                placeholder="Enter toy code"
            )
        
        with col3:
            # 3. Toy selection
            selected_toy = st.selectbox(
                "Select Toy *",
                options=[""] + TOYS,
                help="Select one of the 24 available toys"
            )
        
        with col4:
            # 4. Location state (optional)
            location_state = st.text_input(
                "Location State (Optional)",
                help="Optional location information",
                placeholder="Enter location state"
            )
        
        # Wide Submit button below inputs
        submitted = st.form_submit_button("Submit", type="primary", use_container_width=True)
        
        if submitted:
            # Validation
            errors = []
            
            if not selected_toy or selected_toy == "":
                errors.append("Please select a toy")
            
            if not balls_code or len(balls_code.strip()) < 1:
                errors.append("Balls code must be at least 1 character")
            
            if not toy_code or len(toy_code.strip()) < 1:
                errors.append("Toy code must be at least 1 character")
            
            if errors:
                for error in errors:
                    st.error(error)
            else:
                # Create new row
                new_row = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "toy": selected_toy,
                    "balls_code": balls_code.strip(),
                    "toy_code": toy_code.strip(),
                    "location_state": location_state.strip() if location_state else ""
                }
                
                # Save individual item to backup immediately
                try:
                    save_backup(new_row)
                except Exception as e:
                    st.warning(f"⚠️ Backup save warning: {str(e)}")
                
                # Reload latest data from CSV to get any concurrent submissions from other users
                load_existing_data.clear()  # Clear cache first to get fresh data
                df_latest = load_existing_data()
                
                # Check for duplicates (in case same entry was submitted concurrently)
                # Create a temporary DataFrame with the new row to check for duplicates
                new_df = pd.DataFrame([new_row])
                combined = pd.concat([df_latest, new_df], ignore_index=True)
                
                # Remove duplicates based on all columns
                df_updated = combined.drop_duplicates(
                    subset=["timestamp", "toy", "balls_code", "toy_code", "location_state"], 
                    keep='last'
                )
                
                # Save to CSV (also creates full backup)
                save_data(df_updated)
                
                # Clear cache to force reload on next render so all users see the update
                load_existing_data.clear()
                
                st.success(f"✅ Label saved successfully! (Total: {len(df_updated)} labels)")
                st.rerun()  # Rerun to refresh the page and show updated data from all users
    
    st.markdown("---")
    
    # Display current data
    st.header("Labeled Data")
    
    if len(df) > 0:
        # Show data table
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Download button
        # csv = df.to_csv(index=False).encode('utf-8')
        # st.download_button(
        #     label="📥 Download CSV",
        #     data=csv,
        #     file_name=f"labeled_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        #     mime="text/csv",
        #     use_container_width=True
        # )
        
        # Summary statistics
        with st.expander("📈 View Summary Statistics"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Entries", len(df))
            with col2:
                st.metric("Unique Toys", df["toy"].nunique())
            with col3:
                st.metric("With Location", len(df[df["location_state"] != ""]))
            
            if len(df) > 0:
                st.subheader("Toy Distribution")
                toy_counts = df["toy"].value_counts()
                st.bar_chart(toy_counts)
    else:
        st.info("No labels yet. Start adding labels using the form above!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Stivanborčo Lab - "
        "<a href='https://cz.linkedin.com/in/václav-stibor-a26892293' target='_blank' style='color: gray; text-decoration: none;'>Václav Stibor</a>, "
        "<a href='https://cz.linkedin.com/in/filip-vančo-a675a4288' target='_blank' style='color: gray; text-decoration: none;'>Filip Vančo</a>"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

