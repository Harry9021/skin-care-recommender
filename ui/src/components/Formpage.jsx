import React, { useState, useEffect } from "react";
import "../Styles/Formpage.css";
import image1 from "../Vectors/image5.png";
import image2 from "../Vectors/image6.png";
import { Link, useNavigate } from "react-router-dom";
import { IoMdArrowRoundBack } from "react-icons/io";

const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";

const Formpage = () => {
    const [formData, setFormData] = useState({
        skinType: "",
        concern1: "",
        concern2: "",
        concern3: "",
    });

    // State to store categories fetched from backend
    const [categories, setCategories] = useState({
        skinTypes: [],
        concerns1: [],
        concerns2: [],
        concerns3: []
    });

    const [isLoading, setIsLoading] = useState(false);
    const [isFetchingCategories, setIsFetchingCategories] = useState(true);
    const [apiError, setApiError] = useState("");
    const [successMessage, setSuccessMessage] = useState("");

    const navigate = useNavigate();

    // Fetch available categories from backend on component mount
    useEffect(() => {
        const fetchCategories = async () => {
            try {
                setIsFetchingCategories(true);
                setApiError("");
                const response = await fetch(`${API_BASE_URL}/api/categories`);

                if (!response.ok) {
                    throw new Error(`Failed to fetch categories: ${response.status}`);
                }

                const data = await response.json();

                if (data.status === "success" && data.categories) {
                    // Handle both old format (array of strings) and new format (array of {value, label})
                    const skinTypesRaw = data.categories['skin type'] || [];
                    const concerns1Raw = data.categories['concern'] || [];
                    const concerns2Raw = data.categories['concern 2'] || [];
                    const concerns3Raw = data.categories['concern 3'] || [];

                    // Convert to consistent format: array of {value, label} objects
                    const parseCategories = (items) => {
                        if (items.length === 0) return [];

                        // If already in new format {value, label}
                        if (items[0] && typeof items[0] === 'object' && 'value' in items[0]) {
                            return items;
                        }

                        // If in old format (strings), convert to new format
                        return items.map(item => ({
                            value: item,
                            label: item
                        }));
                    };

                    const skinTypes = parseCategories(skinTypesRaw);
                    const concerns1 = parseCategories(concerns1Raw);
                    const concerns2 = parseCategories(concerns2Raw);
                    const concerns3 = parseCategories(concerns3Raw);

                    setCategories({
                        skinTypes: skinTypes,
                        concerns1: concerns1,
                        concerns2: concerns2,
                        concerns3: concerns3
                    });
                    setSuccessMessage("Categories loaded successfully");
                } else {
                    throw new Error("Invalid response format from server");
                }
            } catch (error) {
                console.error("Error fetching categories:", error);
                setApiError(
                    `Failed to load form options. Please ensure the backend server is running at ${API_BASE_URL}`
                );
            } finally {
                setIsFetchingCategories(false);
            }
        };

        fetchCategories();
    }, []);

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData({
            ...formData,
            [name]: value,
        });
        // Clear error when user makes changes
        if (apiError) setApiError("");
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsLoading(true);
        setApiError("");

        // Validate form
        if (!formData.skinType || !formData.concern1 || !formData.concern2 || !formData.concern3) {
            setApiError("Please fill in all fields");
            setIsLoading(false);
            return;
        }

        // Send request with correct API endpoint
        const userInput = {
            skin_type: formData.skinType,
            concern_1: formData.concern1,
            concern_2: formData.concern2,
            concern_3: formData.concern3,
            top_n: 10
        };

        console.log("Sending request:", userInput);

        try {
            const response = await fetch(`${API_BASE_URL}/api/recommend`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(userInput),
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.message || `HTTP error! status: ${response.status}`);
            }

            if (data.error) {
                setApiError(data.error);
                setIsLoading(false);
                return;
            }

            console.log("Received recommendations:", data.recommendations);

            // Navigate to results page with recommendations
            navigate("/results", {
                state: {
                    recommendations: data.recommendations,
                    userInput: formData
                }
            });

        } catch (error) {
            console.error("Error fetching recommendations:", error);
            setApiError(error.message || "Failed to retrieve recommendations. Please try again.");
        } finally {
            setIsLoading(false);
        }
    };

    // Helper function to get available options for each concern dropdown
    // This prevents selecting the same concern multiple times across different dropdowns
    const getAvailableConcerns1 = () => {
        return categories.concerns1.filter(concern =>
            concern.value != formData.concern2 && concern.value != formData.concern3
        );
    };

    const getAvailableConcerns2 = () => {
        return categories.concerns2.filter(concern =>
            concern.value != formData.concern1 && concern.value != formData.concern3
        );
    };

    const getAvailableConcerns3 = () => {
        return categories.concerns3.filter(concern =>
            concern.value != formData.concern1 && concern.value != formData.concern2
        );
    };

    return (
        <div className="form-image-container">
            <div className="blue-box">
                <Link to="/">
                    <div className="backarrow">
                        <IoMdArrowRoundBack style={{ width: 30, height: 30, marginLeft: 10, marginTop: 10 }} />
                    </div>
                </Link>
            </div>

            <div className="form-container">
                <form onSubmit={handleSubmit}>
                    <h3>Fill Below Information to know your skincare products.</h3>

                    {/* Error Message Display */}
                    {apiError && (
                        <div className="alert alert-danger" role="alert">
                            <span>⚠️ {apiError}</span>
                        </div>
                    )}

                    {/* Success Message Display */}
                    {successMessage && !isFetchingCategories && (
                        <div className="alert alert-success" role="alert">
                            <span>✓ {successMessage}</span>
                        </div>
                    )}

                    {/* Loading State */}
                    {isFetchingCategories && (
                        <div className="loading-spinner">
                            <div className="spinner"></div>
                            <p>Loading form options...</p>
                        </div>
                    )}

                    <label>What is your skin type?</label>
                    <select
                        name="skinType"
                        value={formData.skinType}
                        onChange={handleChange}
                        required
                        disabled={isLoading || isFetchingCategories}
                    >
                        <option value="">Select</option>
                        {categories.skinTypes.map(skinType => (
                            <option key={skinType.value} value={skinType.value}>
                                {skinType.label}
                            </option>
                        ))}
                    </select>

                    <label>Skin concern 1?</label>
                    <select
                        name="concern1"
                        value={formData.concern1}
                        onChange={handleChange}
                        required
                        disabled={isLoading || isFetchingCategories}
                    >
                        <option value="">Select</option>
                        {getAvailableConcerns1().map(concern => (
                            <option key={concern.value} value={concern.value}>
                                {concern.label}
                            </option>
                        ))}
                    </select>

                    <label>Skin concern 2?</label>
                    <select
                        name="concern2"
                        value={formData.concern2}
                        onChange={handleChange}
                        required
                        disabled={isLoading || isFetchingCategories}
                    >
                        <option value="">Select</option>
                        {getAvailableConcerns2().map(concern => (
                            <option key={concern.value} value={concern.value}>
                                {concern.label}
                            </option>
                        ))}
                    </select>

                    <label>Skin concern 3?</label>
                    <select
                        name="concern3"
                        value={formData.concern3}
                        onChange={handleChange}
                        required
                        disabled={isLoading || isFetchingCategories}
                    >
                        <option value="">Select</option>
                        {getAvailableConcerns3().map(concern => (
                            <option key={concern.value} value={concern.value}>
                                {concern.label}
                            </option>
                        ))}
                    </select>

                    <button
                        className="submit-button"
                        type="submit"
                        disabled={isLoading || isFetchingCategories}
                        style={{
                            opacity: (isLoading || isFetchingCategories) ? 0.7 : 1,
                            cursor: (isLoading || isFetchingCategories) ? 'not-allowed' : 'pointer'
                        }}
                    >
                        {isFetchingCategories ? 'Loading...' : isLoading ? 'Getting Recommendations...' : 'Submit'}
                    </button>
                </form>
            </div>

            <div className="imager-x">
                <div>
                    <img className="image-6" src={image2} alt="" />
                </div>
                <div>
                    <img className="image-5" src={image1} alt="" />
                </div>
            </div>
        </div>
    );
};

export default Formpage;