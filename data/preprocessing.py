"""
Data preprocessing module for VAE pipeline.
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Handles data preprocessing for the VAE pipeline.
    """
    
    def __init__(
        self,
        categorical_cols: Optional[List[str]] = None,
        numerical_cols: Optional[List[str]] = None
    ):
        """
        Initialize preprocessor.
        
        Args:
            categorical_cols: List of categorical column names
            numerical_cols: List of numerical column names
        """
        self.categorical_cols = categorical_cols or []
        self.numerical_cols = numerical_cols or []
        
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.scaler = StandardScaler()
        self.categorical_cardinality = {}
        
        self.is_fitted = False
    
    def fit_transform(
        self, 
        df: pd.DataFrame,
        output_dir: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fit preprocessor and transform data.
        
        Args:
            df: Input DataFrame
            output_dir: Directory to save preprocessing objects
            
        Returns:
            Tuple of (preprocessed_df, metadata)
        """
        logger.info("Starting data preprocessing...")
        
        # Apply filtering criteria
        df_filtered = self._apply_filters(df)
        
        # Handle missing values and duplicates
        df_clean = self._clean_data(df_filtered)
        
        # Separate features
        cat_cols, num_cols = self._validate_columns(df_clean)
        
        # Preprocess features
        df_preprocessed = self._preprocess_features(df_clean, cat_cols, num_cols)
        
        # Create metadata
        metadata = self._create_metadata(df, df_filtered, df_clean, df_preprocessed)
        
        # Save preprocessing objects if output directory provided
        if output_dir:
            self._save_preprocessing_objects(output_dir, cat_cols, num_cols)
        
        self.is_fitted = True
        logger.info("Data preprocessing completed successfully!")
        
        return df_preprocessed, metadata
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Apply same cleaning steps
        df_filtered = self._apply_filters(df)
        df_clean = self._clean_data(df_filtered)
        
        # Get existing column lists
        cat_cols = [col for col in self.categorical_cols if col in df_clean.columns]
        num_cols = [col for col in self.numerical_cols if col in df_clean.columns]
        
        # Transform features
        df_preprocessed = self._transform_features(df_clean, cat_cols, num_cols)
        
        return df_preprocessed
    
    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply filtering criteria to the data."""
        logger.info(f"Initial data shape: {df.shape}")
        
        # Remove the first column if it's an index
        if df.columns[0].lower() in ['index', 'unnamed: 0']:
            df = df.iloc[:, 1:]
        
        # Apply specific filters
        original_size = len(df)
        
        # Filter code_country
        if 'code_country' in df.columns:
            df = df[df['code_country'] != 0]
            logger.info(f"Filtered code_country != 0: {len(df)}/{original_size} remaining")
        
        # Filter NumberOfLanesInPrincipalRoad
        if 'NumberOfLanesInPrincipalRoad' in df.columns:
            df = df[df['NumberOfLanesInPrincipalRoad'] <= 5]
            logger.info(f"Filtered lanes <= 5: {len(df)}/{original_size} remaining")
        
        # Filter T1_climate_day_period
        if 'T1_climate_day_period' in df.columns:
            df = df[df['T1_climate_day_period'] != -127]
            logger.info(f"Filtered day_period != -127: {len(df)}/{original_size} remaining")
        
        logger.info(f"Data shape after filtering: {df.shape}")
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values and duplicates."""
        logger.info("Handling missing values and duplicates...")
        
        initial_size = len(df)
        
        # Remove rows with missing values
        df_clean = df.dropna()
        logger.info(f"Removed {initial_size - len(df_clean)} rows with missing values")
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        logger.info(f"Removed {len(df) - len(df_clean)} duplicate rows")
        
        logger.info(f"Data shape after cleaning: {df_clean.shape}")
        return df_clean
    
    def _validate_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Validate and filter column lists."""
        # Check for missing columns
        missing_cat = [col for col in self.categorical_cols if col not in df.columns]
        missing_num = [col for col in self.numerical_cols if col not in df.columns]
        
        if missing_cat:
            logger.warning(f"Missing categorical columns: {missing_cat}")
        if missing_num:
            logger.warning(f"Missing numerical columns: {missing_num}")
        
        # Filter to existing columns
        cat_cols = [col for col in self.categorical_cols if col in df.columns]
        num_cols = [col for col in self.numerical_cols if col in df.columns]
        
        logger.info(f"Using {len(cat_cols)} categorical and {len(num_cols)} numerical columns")
        
        return cat_cols, num_cols
    
    def _preprocess_features(
        self, 
        df: pd.DataFrame, 
        cat_cols: List[str], 
        num_cols: List[str]
    ) -> pd.DataFrame:
        """Preprocess categorical and numerical features."""
        logger.info("Processing categorical and numerical variables...")
        
        # One-hot encode categorical variables
        if cat_cols:
            df_categorical_encoded = self._encode_categorical(df, cat_cols)
        else:
            df_categorical_encoded = pd.DataFrame(index=df.index)
        
        # Normalize numerical variables
        if num_cols:
            df_numerical_scaled = self._scale_numerical(df, num_cols)
        else:
            df_numerical_scaled = pd.DataFrame(index=df.index)
        
        # Combine features
        df_combined = pd.concat([df_numerical_scaled, df_categorical_encoded], axis=1)
        
        logger.info(f"Final preprocessed shape: {df_combined.shape}")
        return df_combined
    
    def _transform_features(
        self, 
        df: pd.DataFrame, 
        cat_cols: List[str], 
        num_cols: List[str]
    ) -> pd.DataFrame:
        """Transform features using fitted preprocessors."""
        # Transform categorical variables
        if cat_cols:
            categorical_data = self.encoder.transform(df[cat_cols])
            encoded_feature_names = self._get_encoded_feature_names(cat_cols)
            df_categorical_encoded = pd.DataFrame(
                categorical_data, 
                columns=encoded_feature_names,
                index=df.index
            )
        else:
            df_categorical_encoded = pd.DataFrame(index=df.index)
        
        # Transform numerical variables
        if num_cols:
            numerical_data = self.scaler.transform(df[num_cols])
            df_numerical_scaled = pd.DataFrame(
                numerical_data, 
                columns=num_cols,
                index=df.index
            )
        else:
            df_numerical_scaled = pd.DataFrame(index=df.index)
        
        # Combine features
        df_combined = pd.concat([df_numerical_scaled, df_categorical_encoded], axis=1)
        
        return df_combined
    
    def _encode_categorical(self, df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
        """Encode categorical variables."""
        logger.info("One-hot encoding categorical variables...")
        
        # Fit and transform
        categorical_data = self.encoder.fit_transform(df[cat_cols])
        
        # Get feature names
        encoded_feature_names = self._get_encoded_feature_names(cat_cols)
        
        df_categorical_encoded = pd.DataFrame(
            categorical_data, 
            columns=encoded_feature_names,
            index=df.index
        )
        
        # Create categorical cardinality mapping
        self._create_categorical_cardinality(cat_cols)
        
        return df_categorical_encoded
    
    def _get_encoded_feature_names(self, cat_cols: List[str]) -> List[str]:
        """Get encoded feature names."""
        encoded_feature_names = []
        for i, feature in enumerate(cat_cols):
            feature_categories = self.encoder.categories_[i]
            for category in feature_categories:
                encoded_feature_names.append(f"{feature}_{category}")
        
        return encoded_feature_names
    
    def _create_categorical_cardinality(self, cat_cols: List[str]) -> None:
        """Create categorical cardinality mapping."""
        self.categorical_cardinality = {}
        start_idx = len(self.numerical_cols)  # Start after numerical columns
        
        for i, feature in enumerate(cat_cols):
            categories = list(self.encoder.categories_[i])
            cardinality = len(categories)
            end_idx = start_idx + cardinality
            
            self.categorical_cardinality[feature] = {
                'cardinality': cardinality,
                'categories': categories,
                'start_idx': start_idx,
                'end_idx': end_idx
            }
            
            start_idx = end_idx
    
    def _scale_numerical(self, df: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
        """Scale numerical variables."""
        logger.info("Normalizing numerical variables...")
        
        numerical_data = self.scaler.fit_transform(df[num_cols])
        df_numerical_scaled = pd.DataFrame(
            numerical_data, 
            columns=num_cols,
            index=df.index
        )
        
        return df_numerical_scaled
    
    def _create_metadata(
        self, 
        df_original: pd.DataFrame,
        df_filtered: pd.DataFrame,
        df_clean: pd.DataFrame,
        df_preprocessed: pd.DataFrame
    ) -> Dict[str, Any]:
        """Create preprocessing metadata."""
        return {
            'original_shape': df_original.shape,
            'filtered_shape': df_filtered.shape,
            'clean_shape': df_clean.shape,
            'final_shape': df_preprocessed.shape,
            'categorical_cols': [col for col in self.categorical_cols if col in df_clean.columns],
            'numerical_cols': [col for col in self.numerical_cols if col in df_clean.columns],
            'categorical_cardinality': self.categorical_cardinality,
            'n_features_original': df_original.shape[1],
            'n_features_final': df_preprocessed.shape[1],
            'n_samples_original': df_original.shape[0],
            'n_samples_final': df_preprocessed.shape[0]
        }
    
    def _save_preprocessing_objects(
        self, 
        output_dir: str, 
        cat_cols: List[str], 
        num_cols: List[str]
    ) -> None:
        """Save preprocessing objects for later use."""
        os.makedirs(output_dir, exist_ok=True)
        
        preprocessing_objects = {
            'encoder': self.encoder,
            'scaler': self.scaler,
            'categorical_cardinality': self.categorical_cardinality,
            'cat_cols': cat_cols,
            'num_cols': num_cols
        }
        
        with open(os.path.join(output_dir, 'preprocessing_objects.pkl'), 'wb') as f:
            pickle.dump(preprocessing_objects, f)
        
        logger.info(f"Preprocessing objects saved to {output_dir}")
    
    @classmethod
    def load_from_file(cls, objects_path: str) -> 'DataPreprocessor':
        """
        Load preprocessor from saved objects.
        
        Args:
            objects_path: Path to preprocessing_objects.pkl file
            
        Returns:
            Loaded DataPreprocessor instance
        """
        with open(objects_path, 'rb') as f:
            objects = pickle.load(f)
        
        preprocessor = cls(
            categorical_cols=objects['cat_cols'],
            numerical_cols=objects['num_cols']
        )
        
        preprocessor.encoder = objects['encoder']
        preprocessor.scaler = objects['scaler']
        preprocessor.categorical_cardinality = objects['categorical_cardinality']
        preprocessor.is_fitted = True
        
        return preprocessor

def preprocess_data(
    data_file: str,
    output_dir: str,
    categorical_cols: Optional[List[str]] = None,
    numerical_cols: Optional[List[str]] = None,
    save_intermediate: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Main preprocessing function.
    
    Args:
        data_file: Path to input data file
        output_dir: Output directory for processed data
        categorical_cols: List of categorical column names
        numerical_cols: List of numerical column names
        save_intermediate: Whether to save intermediate files
        
    Returns:
        Tuple of (original_df, preprocessed_df, metadata)
    """
    # Check if data file exists
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    # Load data
    logger.info(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    
    # Create preprocessor
    preprocessor = DataPreprocessor(categorical_cols, numerical_cols)
    
    # Fit and transform
    df_preprocessed, metadata = preprocessor.fit_transform(df, output_dir)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save preprocessed data
    preprocessed_file = os.path.join(output_dir, 'preprocessed_data.csv')
    df_preprocessed.to_csv(preprocessed_file, index=False)
    logger.info(f"Preprocessed data saved to {preprocessed_file}")
    
    if save_intermediate:
        # Save filtered data
        df_filtered = preprocessor._apply_filters(df)
        df_clean = preprocessor._clean_data(df_filtered)
        
        filtered_file = os.path.join(output_dir, 'filtered_data.csv')
        df_clean.to_csv(filtered_file, index=False)
        logger.info(f"Filtered data saved to {filtered_file}")
        
        # Save metadata
        metadata_file = os.path.join(output_dir, 'preprocessing_metadata.pkl')
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"Metadata saved to {metadata_file}")
    
    logger.info("Data preprocessing pipeline completed!")
    logger.info(f"Final shape: {df_preprocessed.shape}")
    
    return df, df_preprocessed, metadata