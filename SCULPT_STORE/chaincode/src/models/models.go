/*
   This file implements the Storage Layer's Smart Contract.
   Copyright (C) 2021  Stavroulla Koumou

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see https://www.gnu.org/licenses/gpl-3.0.html.
*/
package main

import (
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/hyperledger/fabric-chaincode-go/shim"
	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

// SimpleChaincode implements the fabric-contract-api-go programming model
type SimpleChaincode struct {
	contractapi.Contract
}

//go:generate counterfeiter -o mocks/transaction.go -fake-name TransactionContext . transactionContext
type transactionContext interface {
	contractapi.TransactionContextInterface
}

//go:generate counterfeiter -o mocks/chaincodestub.go -fake-name ChaincodeStub . chaincodeStub
type chaincodeStub interface {
	shim.ChaincodeStubInterface
}

//go:generate counterfeiter -o mocks/statequeryiterator.go -fake-name StateQueryIterator . stateQueryIterator
type stateQueryIterator interface {
	shim.StateQueryIteratorInterface
}

type MetaData struct {
	ModelID               string `json:"model_id"`
	Tag1                  string `json:"tag1"`
	Tag2                  string `json:"tag2"`
	SerializationEncoding string `json:"serialization_encoding"`
	PageNumber            int    `json:"page_number"`
}

type GenericPage struct {
	ModelID   string `json:"model_id"`
	PageID    int    `json:"page_id"`
	PageBytes int    `json:"page_bytes"` //data+digest
	Data      string `json:"data"`
	Digest    string `json:"digest"`
}

type GenericPageWrapper struct {
	Key   string      `json: key`
	Value GenericPage `json: value`
}

// PaginatedQueryResult structure used for returning paginated query results and metadata
type PaginatedQueryResult struct {
	Records             []*GenericPageWrapper `json:"records"`
	FetchedRecordsCount int32                 `json:"fetchedRecordsCount"`
	Bookmark            string                `json:"bookmark"`
}

type MetaPaginatedQueryResult struct {
	Records             []*MetaData `json:"records"`
	FetchedRecordsCount int32       `json:"fetchedRecordsCount"`
	Bookmark            string      `json:"bookmark"`
}

////////////////////////////////////////////////////////////////////////////////////////////////

// Creates a new client token using uuid
func (t *SimpleChaincode) CreateClientToken(ctx contractapi.TransactionContextInterface, token string) (string, error) {

	err := ctx.GetStub().PutState(token, []byte{'t', 'o', 'k', 'e', 'n'})
	if err != nil {
		return "", err
	}

	return token, nil
}

 func NewSimpleChaincode(contract contractapi.Contract) *js.Object {
	return js.MakeWrapper(&SimpleChaincode{contract})
  }
// Checks if a client token exists
func (t *SimpleChaincode) CheckClientToken(ctx contractapi.TransactionContextInterface, token string) (bool, error) {
	tokenBytes, err := ctx.GetStub().GetState(token)
	if err != nil {
		return false, fmt.Errorf("failed to retrieve token %s from world state. %v", token, err)
	}

	return tokenBytes != nil, nil
}

////////////////////////////////////////////////////////////////////////////////////////////////

// Submits a metadata entry to the ledger.
// Metadata entries have to do with the model's page metadata.
func (t *SimpleChaincode) SubmitMeta(ctx contractapi.TransactionContextInterface, token, modelID, entryString string) error {

	tokenBytes, err := ctx.GetStub().GetState(token)
	if err != nil {
		return fmt.Errorf("failed to retrieve token %s from world state. %v", token, err)
	}

	if tokenBytes == nil {
		return fmt.Errorf("authorization not granded: token %s is invalid", token)
	}

	var entry MetaData
	err = json.Unmarshal([]byte(entryString), &entry)
	if err != nil {
		return fmt.Errorf("failed to process json data; ensure correct structure")
	}

	entryBytes, err := json.Marshal(entry)
	if err != nil {
		return err
	}

	err = ctx.GetStub().PutState(modelID+"_metadata", entryBytes)
	if err != nil {
		return err
	}

	return nil
}

// Submits a page-data entry to the ledger.
// Page-data entries have to do with the model's actual data.
// PpageType can be either of the following: model, weights, initialization, checkpoints
func (t *SimpleChaincode) SubmitPage(ctx contractapi.TransactionContextInterface, token, modelID, entryString string) error {

	tokenBytes, err := ctx.GetStub().GetState(token)
	if err != nil {
		return fmt.Errorf("failed to retrieve token %s from world state. %v", token, err)
	}

	if tokenBytes == nil {
		return fmt.Errorf("authorization not granded: token %s is invalid", token)
	}

	var entry GenericPage
	err = json.Unmarshal([]byte(entryString), &entry)
	if err != nil {
		return fmt.Errorf("failed to process json data; ensure correct structure")
	}

	entryBytes, err := json.Marshal(entry)
	if err != nil {
		return err
	}

	err = ctx.GetStub().PutState(modelID+"_"+strconv.Itoa(entry.PageID), entryBytes)
	if err != nil {
		return err
	}

	return nil
}

// QueryMetaData retrieves the metadata of a model from the ledger
func (t *SimpleChaincode) QueryMetaData(ctx contractapi.TransactionContextInterface, token, modelID string) (*MetaData, error) {

	tokenBytes, err := ctx.GetStub().GetState(token)
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve token %s from world state. %v", token, err)
	}

	if tokenBytes == nil {
		return nil, fmt.Errorf("authorization not granded: token %s is invalid", token)
	}

	//retrieve the model's metadata
	metaBytes, err := ctx.GetStub().GetState(modelID + "_metadata")
	if err != nil {
		return nil, fmt.Errorf("failed to get metadata for model %s: %v", modelID, err)
	}
	if metaBytes == nil {
		return nil, fmt.Errorf("model_id %s does not exist", modelID)
	}

	var meta MetaData
	err = json.Unmarshal(metaBytes, &meta)
	if err != nil {
		return nil, err
	}

	return &meta, nil
}

////////////////////////////////////////////////////////////////////////////////////////////////
// Performs an adhoc query on MetaData ledger entries
func (t *SimpleChaincode) QueryMetaDataWithPagination(ctx contractapi.TransactionContextInterface, token, queryString string, pageSize int, bookmark string) (*MetaPaginatedQueryResult, error) {

	tokenBytes, err := ctx.GetStub().GetState(token)
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve token %s from world state. %v", token, err)
	}

	if tokenBytes == nil {
		return nil, fmt.Errorf("authorization not granded: token %s is invalid", token)
	}

	return getMetaQueryResultForQueryStringWithPagination(ctx, queryString, int32(pageSize), bookmark)
}

func getMetaQueryResultForQueryStringWithPagination(ctx contractapi.TransactionContextInterface, queryString string, pageSize int32, bookmark string) (*MetaPaginatedQueryResult, error) {

	resultsIterator, responseMetadata, err := ctx.GetStub().GetQueryResultWithPagination(queryString, pageSize, bookmark)
	if err != nil {
		return nil, fmt.Errorf("here 3") //err
	}
	defer resultsIterator.Close()

	metas, err := constructMetaQueryResponseFromIterator(resultsIterator)
	if err != nil {
		return nil, err
	}

	if metas == nil {
		return nil, nil
	}

	return &MetaPaginatedQueryResult{
		Records:             metas,
		FetchedRecordsCount: responseMetadata.FetchedRecordsCount,
		Bookmark:            responseMetadata.Bookmark,
	}, nil
}

// constructQueryResponseFromIterator constructs a slice of assets from the resultsIterator
func constructMetaQueryResponseFromIterator(resultsIterator shim.StateQueryIteratorInterface) ([]*MetaData, error) {

	var metas []*MetaData
	for resultsIterator.HasNext() {
		queryResult, err := resultsIterator.Next()
		if err != nil {
			return nil, fmt.Errorf("here 1") //err
		}

		var meta MetaData
		err = json.Unmarshal(queryResult.Value, &meta)
		if err != nil {
			return nil, fmt.Errorf("here 2") //err
		}

		metas = append(metas, &meta)
	}

	return metas, nil
}

////////////////////////////////////////////////////////////////////////////////////////////////

// Returns the ledger entries which correspond to a specific model
func (t *SimpleChaincode) QueryLatestModelWithPagination(ctx contractapi.TransactionContextInterface, token, modelID string, pageSize int, bookmark string) (*PaginatedQueryResult, error) {

	tokenBytes, err := ctx.GetStub().GetState(token)
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve token %s from world state. %v", token, err)
	}

	if tokenBytes == nil {
		return nil, fmt.Errorf("authorization not granded: token %s is invalid", token)
	}

	//retrieve the model's metadata
	metaBytes, err := ctx.GetStub().GetState(modelID + "_metadata")
	if err != nil {
		return nil, fmt.Errorf("failed to get metadata for model %s: %v", modelID, err)
	}
	if metaBytes == nil {
		return nil, fmt.Errorf("model_id %s does not exist", modelID)
	}

	var meta MetaData
	err = json.Unmarshal(metaBytes, &meta)
	if err != nil {
		return nil, err
	}

	queryString := fmt.Sprintf(`{"selector":{"$and":[{"model_id":"%s"},{"page_id": {"$lt":%d}}]}}`, modelID, meta.PageNumber)
	return getQueryResultForQueryStringWithPagination(ctx, queryString, int32(pageSize), bookmark)
}

func getQueryResultForQueryStringWithPagination(ctx contractapi.TransactionContextInterface, queryString string, pageSize int32, bookmark string) (*PaginatedQueryResult, error) {

	resultsIterator, responseMetadata, err := ctx.GetStub().GetQueryResultWithPagination(queryString, pageSize, bookmark)
	if err != nil {
		return nil, err
	}
	defer resultsIterator.Close()

	pages, err := constructQueryResponseFromIterator(resultsIterator)
	if err != nil {
		return nil, err
	}

	if pages == nil {
		return nil, nil
	}

	return &PaginatedQueryResult{
		Records:             pages,
		FetchedRecordsCount: responseMetadata.FetchedRecordsCount,
		Bookmark:            responseMetadata.Bookmark,
	}, nil
}

// constructQueryResponseFromIterator constructs a slice of assets from the resultsIterator
func constructQueryResponseFromIterator(resultsIterator shim.StateQueryIteratorInterface) ([]*GenericPageWrapper, error) {

	var pages []*GenericPageWrapper
	for resultsIterator.HasNext() {
		queryResult, err := resultsIterator.Next()
		if err != nil {
			return nil, err
		}

		var page GenericPage
		err = json.Unmarshal(queryResult.Value, &page)
		if err != nil {
			return nil, err
		}

		var wrap = GenericPageWrapper{
			Key:   strconv.Itoa(page.PageID),
			Value: page,
		}

		pages = append(pages, &wrap)
	}

	return pages, nil
}

////////////////////////////////////////////////////////////////////////////////////////////////

// DeleteKeys deletes the keys in the given array
func (t *SimpleChaincode) DeleteKeys(ctx contractapi.TransactionContextInterface, token string, keys []string) error {

	tokenBytes, err := ctx.GetStub().GetState(token)
	if err != nil {
		return fmt.Errorf("failed to retrieve token %s from world state. %v", token, err)
	}

	if tokenBytes == nil {
		return fmt.Errorf("authorization not granded: token %s is invalid", token)
	}

	for _, key := range keys {
		err = ctx.GetStub().DelState(key)
		if err != nil {
			return fmt.Errorf("failed to delete ledger entry %s: %v", key, err)
		}
	}

	return nil
}

// GetKeys returns all the keys that much the provided adhoc
func (t *SimpleChaincode) GetKeys(ctx contractapi.TransactionContextInterface, token, modelID, queryString string, pageSize int, bookmark string) ([]string, error) {

	var keys []string

	tokenBytes, err := ctx.GetStub().GetState(token)
	if err != nil {
		return keys, fmt.Errorf("failed to retrieve token %s from world state. %v", token, err)
	}

	if tokenBytes == nil {
		return keys, fmt.Errorf("authorization not granded: token %s is invalid", token)
	}

	//retrieve the model's metadata
	metaBytes, err := ctx.GetStub().GetState(modelID + "_metadata")
	if err != nil {
		return keys, fmt.Errorf("failed to get metadata for model %s: %v", modelID, err)
	}
	if metaBytes == nil {
		return keys, fmt.Errorf("model_id %s does not exist", modelID)
	}

	book := bookmark

	for true {
		resultsIterator, responseMetadata, err := ctx.GetStub().GetQueryResultWithPagination(queryString, int32(pageSize), book)
		if err != nil {
			return keys, err
		}
		defer resultsIterator.Close()

		for resultsIterator.HasNext() {
			queryResult, err := resultsIterator.Next()
			if err != nil {
				return keys, err
			}

			keys = append(keys, queryResult.Key)

		}

		if responseMetadata.FetchedRecordsCount == 0 {
			break
		}

		book = responseMetadata.Bookmark

	}

	return keys, nil
}

func (t *SimpleChaincode) GetCleanUpKeys(ctx contractapi.TransactionContextInterface, token, modelID string, pageSize int, bookmark string) ([]string, error) {

	var keys []string

	tokenBytes, err := ctx.GetStub().GetState(token)
	if err != nil {
		return keys, fmt.Errorf("failed to retrieve token %s from world state. %v", token, err)
	}

	if tokenBytes == nil {
		return keys, fmt.Errorf("authorization not granded: token %s is invalid", token)
	}

	//retrieve the model's metadata
	metaBytes, err := ctx.GetStub().GetState(modelID + "_metadata")
	if err != nil {
		return keys, fmt.Errorf("failed to get metadata for model %s: %v", modelID, err)
	}
	if metaBytes == nil {
		return keys, fmt.Errorf("model_id %s does not exist", modelID)
	}

	var meta MetaData
	err = json.Unmarshal(metaBytes, &meta)
	if err != nil {
		return nil, err
	}

	queryString := fmt.Sprintf(`{"selector":{"$and":[{"model_id":"%s"},{"page_id": {"$gte":%d}}]}}`, modelID, meta.PageNumber)

	book := bookmark

	for true {
		resultsIterator, responseMetadata, err := ctx.GetStub().GetQueryResultWithPagination(queryString, int32(pageSize), book)
		if err != nil {
			return keys, err
		}
		defer resultsIterator.Close()

		for resultsIterator.HasNext() {
			queryResult, err := resultsIterator.Next()
			if err != nil {
				return keys, err
			}

			keys = append(keys, queryResult.Key)

		}

		if responseMetadata.FetchedRecordsCount == 0 {
			break
		}

		book = responseMetadata.Bookmark

	}

	return keys, nil
}

////////////////////////////////////////////////////////////////////////////////////////////////

func (t *SimpleChaincode) Init(ctx contractapi.TransactionContextInterface) error {

	err := ctx.GetStub().PutState("token", []byte{'t', 'o', 'k', 'e', 'n'})
	if err != nil {
		return err
	}

	return nil
}

func main() {

	sc := new(SimpleChaincode)

	cc, err := contractapi.NewChaincode(sc)

	if err != nil {
		panic(err.Error())
	}

	if err := cc.Start(); err != nil {
		panic(err.Error())
	}

}
