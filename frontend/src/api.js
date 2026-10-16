import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000/api",
  timeout: 15000,
});

export function errorMessage(error) {
  return error.response?.data?.detail || error.message || "The API request failed.";
}

export default api;
