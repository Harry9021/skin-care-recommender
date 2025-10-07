import React, { useState, useEffect } from "react";
import "../Styles/Formpage.css";
import image1 from "../Vectors/image5.png";
import image2 from "../Vectors/image6.png";
import { Link, useNavigate } from "react-router-dom";
import { IoMdArrowRoundBack } from "react-icons/io";

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
        concerns1: [],  // Separate array for concern 1
        concerns2: [],  // Separate array for concern 2
        concerns3: []   // Separate array for concern 3
    });

    const [isLoading, setIsLoading] = useState(false);
    const [isFetchingCategories, setIsFetchingCategories] = useState(true);
    const [error, setError] = useState("");

    const navigate = useNavigate();

    // Fetch available categories from backend on component mount
    useEffect(() => {
        const fetchCategories = async () => {
            try {
                setIsFetchingCategories(true);
                const response = await fetch("http://localhost:5000/categories");

                if (!response.ok) {
                    throw new Error(`Failed to fetch categories: ${response.status}`);
                }

                const data = await response.json();

                if (data.status === "success" && data.categories) {
                    // Extract skin types from the backend response
                    const skinTypes = data.categories['skin type'] || [];

                    // Keep concerns separate for each dropdown
                    // Remove duplicates within each concern category
                    const concerns1 = [...new Set(data.categories['concern'] || [])].sort();
                    const concerns2 = [...new Set(data.categories['concern 2'] || [])].sort();
                    const concerns3 = [...new Set(data.categories['concern 3'] || [])].sort();

                    setCategories({
                        skinTypes: skinTypes,
                        concerns1: concerns1,
                        concerns2: concerns2,
                        concerns3: concerns3
                    });

                } else {
                    throw new Error("Invalid response format from server");
                }
            } catch (error) {
                console.error("Error fetching categories:", error);
                setError(
                    `Failed to load form options. Please ensure the backend server is running at http://localhost:5000`
                );
            } finally {
                setIsFetchingCategories(false);
            }
        };

        fetchCategories();
    }, []); // Empty dependency array means this runs once on mount

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData({
            ...formData,
            [name]: value,
        });
        // Clear error when user makes changes
        if (error) setError("");
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsLoading(true);
        setError("");

        // Send the exact values from the dropdown (no transformation needed)
        const userInput = {
            skin_type: formData.skinType,
            concern_1: formData.concern1,
            concern_2: formData.concern2,
            concern_3: formData.concern3,
            top_n: 10
        };

        console.log("Sending request:", userInput);

        try {
            const response = await fetch(
                "http://localhost:5000/recommend",
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify(userInput),
                }
            );

            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || `HTTP error! status: ${response.status}`);
            }

            if (data.error) {
                setError(data.error);
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
            setError(error.message || "Failed to retrieve recommendations. Please try again.");
        } finally {
            setIsLoading(false);
        }
    };

    // Helper function to get available options for each concern dropdown
    // This prevents selecting the same concern multiple times across different dropdowns
    const getAvailableConcerns1 = () => {
        return categories.concerns1.filter(concern =>
            concern !== formData.concern2 && concern !== formData.concern3
        );
    };

    const getAvailableConcerns2 = () => {
        return categories.concerns2.filter(concern =>
            concern !== formData.concern1 && concern !== formData.concern3
        );
    };

    const getAvailableConcerns3 = () => {
        return categories.concerns3.filter(concern =>
            concern !== formData.concern1 && concern !== formData.concern2
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
                    {error && (
                        <div style={{
                            backgroundColor: '#fee',
                            color: '#c33',
                            padding: '10px',
                            borderRadius: '5px',
                            marginBottom: '15px',
                            border: '1px solid #fcc',
                            fontSize: '14px'
                        }}>
                            {error}
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
                            <option key={skinType} value={skinType}>
                                {skinType}
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
                            <option key={concern} value={concern}>
                                {concern}
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
                            <option key={concern} value={concern}>
                                {concern}
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
                            <option key={concern} value={concern}>
                                {concern}
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